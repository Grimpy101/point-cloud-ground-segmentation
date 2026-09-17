use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    time::Instant,
};

use clap::Parser;
use itertools::izip;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::point::Point;

mod io;
mod math;
mod point;

/// Small tool for ground segmentation
#[derive(clap::Parser)]
pub struct Args {
    /// Input point cloud file (LAS/LAZ)
    #[arg(short, long)]
    input: PathBuf,
    /// Output LAS/LAZ file
    #[arg(short, long)]
    output: PathBuf,
    /// Grid resolution for initial ground sampling
    #[arg(short, long)]
    cell_size: f64,
    /// Factor of how high should the ground be considered
    #[arg(short, long)]
    elevation_threshold: f64,
    /// Perform test against GT
    #[arg(short, long)]
    test: bool,
}

fn sample_ground_points(points: &[Point], grid_cell_size: f64) -> Vec<usize> {
    let mut cells: HashMap<(i64, i64), usize> = HashMap::new();

    for (i, p) in points.iter().enumerate() {
        let ix = (p.x / grid_cell_size).round() as i64;
        let iy = (p.y / grid_cell_size).round() as i64;
        let key = (ix, iy);

        cells
            .entry(key)
            .and_modify(|best_idx| {
                if p.z < points[*best_idx].z {
                    *best_idx = i;
                }
            })
            .or_insert(i);
    }

    cells.into_values().collect()
}

fn compute_normals(points: &mut [Point], kdtree: &kdtree::KDTree, k: usize) {
    let normals: Vec<[f64; 3]> = points
        .par_iter()
        .map(|point| {
            let neighbourhood = kdtree.k_nearest(point, k, points);
            let n = neighbourhood.len() as f64;

            let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
            for nb in neighbourhood.iter() {
                let p = &points[nb.index];
                cx += p.x;
                cy += p.y;
                cz += p.z;
            }
            cx /= n;
            cy /= n;
            cz /= n;

            let mut cov = [[0.0; 3]; 3];
            for nb in neighbourhood.iter() {
                let p = &points[nb.index];
                let d = [p.x - cx, p.y - cy, p.z - cz];
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] += d[i] * d[j];
                    }
                }
            }
            for row in cov.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }

            math::smallest_eigenvector(cov)
        })
        .collect();

    points
        .par_iter_mut()
        .zip(normals.par_iter())
        .for_each(|(point, normal)| {
            point.nx = normal[0];
            point.ny = normal[1];
            point.nz = normal[2];
        });
}

fn propagate_ground(
    points: &mut [Point],
    kdtree: &kdtree::KDTree,
    sample_ground: &mut [usize],
    threshold: f64,
    k: usize,
) {
    let mut queue = sample_ground.iter().copied().collect::<VecDeque<usize>>();
    let mut queued = queue.iter().copied().collect::<HashSet<usize>>();

    while let Some(idx) = queue.pop_front() {
        let query_point = points[idx].clone();
        let neighbourhood = kdtree.k_nearest(&query_point, k, points);

        if neighbourhood.len() < 3 {
            continue;
        }

        for n in neighbourhood.iter() {
            let ni = n.index;
            if points[ni].is_ground {
                continue;
            }

            let dx = points[ni].x - query_point.x;
            let dy = points[ni].y - query_point.y;
            let dz = points[ni].z - query_point.z;
            let len = (dx * dx + dy * dy + dz * dz).sqrt();

            let normal_condition = if len >= 1e-9 {
                let dot = (dx * points[ni].nx + dy * points[ni].ny + dz * points[ni].nz) / len;
                dot.abs() < 0.2
            } else {
                true
            };

            if dz < threshold && normal_condition {
                points[ni].is_ground = true;
                if queued.insert(ni) {
                    queue.push_back(ni);
                }
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    if !args.input.exists() {
        panic!("File {} does not exist", args.input.to_string_lossy());
    }
    let k = 10;

    // ------------- Point cloud reading ------------- //

    let start_time = Instant::now();
    let (mut points, classes, transforms) = io::read_las(&args.input, args.test);
    let end_time = Instant::now();

    println!(
        "{} points read ({} s)",
        points.len(),
        (end_time - start_time).as_secs_f32()
    );

    // ------------- Tree building ------------- //

    let start_time = Instant::now();
    let tree = kdtree::KDTree::build(&points);
    let end_time = Instant::now();

    println!(
        "Built a KDTree ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    // ------------- Normal computation ------------- //

    let start_time = Instant::now();
    compute_normals(&mut points, &tree, k);
    let end_time = Instant::now();

    println!(
        "Computed normals ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    // ------------- Initial ground sampling ------------- //

    let start_time = Instant::now();
    let mut ground_point_indices = sample_ground_points(&points, args.cell_size);
    let end_time = Instant::now();

    println!(
        "Sampled {} ground points ({} s)",
        ground_point_indices.len(),
        (end_time - start_time).as_secs_f32()
    );

    let threshold = args.elevation_threshold;

    // ------------- Ground filling ------------- //

    let start_time = Instant::now();
    propagate_ground(&mut points, &tree, &mut ground_point_indices, threshold, k);
    let end_time = Instant::now();

    println!(
        "Created ground! ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    // ------------- Testing ------------- //

    if args.test {
        let mut tps = 0.0;
        let mut fps = 0.0;
        let mut fns = 0.0;
        for (p, c) in izip!(points.iter_mut(), classes.iter()) {
            if p.is_ground {
                if *c == 2 {
                    tps += 1.0;
                } else {
                    fps += 1.0;
                    p.wrong = 1;
                }
            } else {
                if *c == 2 {
                    fns += 1.0;
                    p.wrong = 2;
                }
            }
        }
        let f1 = (2.0 * tps) / (2.0 * tps + fps + fns);
        println!("F1: {}, TP: {}, FP: {}, FN: {}", f1, tps, fps, fns);
    }

    // ------------- Output ------------- //

    io::write_laz(&points, transforms, &args.output).unwrap();
}
