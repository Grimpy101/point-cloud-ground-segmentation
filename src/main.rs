use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::Parser;
use itertools::izip;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

#[derive(Clone)]
pub struct Point {
    x: f64,
    y: f64,
    z: f64,
    nx: f64,
    ny: f64,
    nz: f64,
    is_ground: bool,
    wrong: u8,
}

impl kdtree::spatial::Spatial for Point {
    fn axis(&self, axis: u8) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => 0.0,
        }
    }
}

#[derive(clap::Parser)]
pub struct Args {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short, long)]
    cell_size: f64,
}

fn smooth(histogram: &[usize], window: usize) -> Vec<f64> {
    let half = window / 2;
    (0..histogram.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(histogram.len());
            histogram[lo..hi].iter().sum::<usize>() as f64 / (hi - lo) as f64
        })
        .collect()
}

fn find_largest_peak(histogram: &[f64]) -> usize {
    let mut best: Option<usize> = None;

    for i in 1..histogram.len() - 1 {
        if histogram[i] > histogram[i - 1]
            && histogram[i] >= histogram[i + 1]
            && best.is_none_or(|b| histogram[i] > histogram[b])
        {
            best = Some(i);
        }
    }

    best.unwrap_or_else(|| {
        histogram
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    })
}

fn find_next_valley(histogram: &[f64], start: usize) -> usize {
    for i in (start + 1)..histogram.len() - 1 {
        if histogram[i] < histogram[i - 1] && histogram[i] <= histogram[i + 1] {
            return i;
        }
    }
    histogram.len() - 1
}

fn elevation_threshold(points: &[Point], kdtree: &kdtree::KDTree, k: usize) -> f64 {
    let num_bins = 10000;

    let offsets: Vec<f64> = points
        .par_iter()
        .map(|point| {
            let neighbourhood = kdtree.k_nearest(point, k, points);
            if neighbourhood.is_empty() {
                return 0.0;
            }
            let n = neighbourhood.len() as f64;
            let mean_z: f64 = neighbourhood
                .iter()
                .map(|nb| points[nb.index].z)
                .sum::<f64>()
                / n;
            (point.z - mean_z).abs()
        })
        .collect();

    let min_offset = offsets.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_offset = offsets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_offset <= min_offset {
        return min_offset;
    }

    let bin_width = (max_offset - min_offset) / num_bins as f64;
    let mut histogram = vec![0usize; num_bins];

    for &offset in &offsets {
        let bin = (((offset - min_offset) / bin_width) as usize).min(num_bins - 1);
        histogram[bin] += 1;
    }

    let histogram = smooth(&histogram, 10);

    let first_peak = find_largest_peak(&histogram);
    let valley_bin = find_next_valley(&histogram, first_peak);
    min_offset + (valley_bin as f64 + 0.5) * bin_width
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

            smallest_eigenvector(cov)
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

fn smallest_eigenvector(m: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = m[0][1].powi(2) + m[0][2].powi(2) + m[1][2].powi(2);

    if p1 < 1e-12 {
        let mut idx = 0;
        for i in 1..3 {
            if m[i][i] < m[idx][idx] {
                idx = i;
            }
        }
        let mut v = [0.0; 3];
        v[idx] = 1.0;
        return v;
    }

    let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    let p2 = (m[0][0] - q).powi(2) + (m[1][1] - q).powi(2) + (m[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = (m[i][j] - if i == j { q } else { 0.0 }) / p;
        }
    }

    let det_b = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);

    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3;
    let smallest = eig1.min(eig2).min(eig3);

    let a = [
        [m[0][0] - smallest, m[0][1], m[0][2]],
        [m[1][0], m[1][1] - smallest, m[1][2]],
        [m[2][0], m[2][1], m[2][2] - smallest],
    ];

    let mut v = cross(a[0], a[1]);
    if norm_sq(v) < 1e-9 {
        v = cross(a[0], a[2]);
    }
    if norm_sq(v) < 1e-9 {
        v = cross(a[1], a[2]);
    }
    normalize(v)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = norm_sq(v).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0] // degenerate fallback
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
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

pub fn write_ply<P: AsRef<Path>>(points: &[Point], path: P) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);

    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "element vertex {}", points.len())?;
    writeln!(writer, "property double x")?;
    writeln!(writer, "property double y")?;
    writeln!(writer, "property double z")?;
    writeln!(writer, "property uchar ground")?;
    writeln!(writer, "property uchar wrong")?;
    writeln!(writer, "end_header")?;

    for p in points {
        writer.write_all(&p.x.to_le_bytes())?;
        writer.write_all(&p.y.to_le_bytes())?;
        writer.write_all(&p.z.to_le_bytes())?;
        writer.write_all(&[p.is_ground as u8])?;
        writer.write_all(&[p.wrong])?;
    }

    writer.flush()?;
    Ok(())
}

fn main() {
    let args = Args::parse();
    if !args.input.exists() {
        panic!("File {} does not exist", args.input.to_string_lossy());
    }
    let k = 10;

    let start_time = Instant::now();
    let mut reader = las::Reader::from_path(args.input).unwrap();
    let n = reader.header().number_of_points() as usize;
    let mut points = Vec::with_capacity(n);
    let mut classes = Vec::with_capacity(n);

    let point_data = reader.read_all().unwrap();
    for (x, y, z, cls) in izip!(
        point_data.x(),
        point_data.y(),
        point_data.z(),
        point_data.classification()
    ) {
        points.push(Point {
            x,
            y,
            z,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            is_ground: false,
            wrong: 0,
        });
        classes.push(cls);
    }
    let end_time = Instant::now();

    println!(
        "{} points read ({} s)",
        points.len(),
        (end_time - start_time).as_secs_f32()
    );

    let start_time = Instant::now();
    let tree = kdtree::KDTree::build(&points);
    let end_time = Instant::now();

    println!(
        "Built a KDTree ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    let start_time = Instant::now();
    compute_normals(&mut points, &tree, k);
    let end_time = Instant::now();

    println!(
        "Computed normals ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    let start_time = Instant::now();
    let mut ground_point_indices = sample_ground_points(&points, args.cell_size);
    let end_time = Instant::now();

    println!(
        "Sampled {} ground points ({} s)",
        ground_point_indices.len(),
        (end_time - start_time).as_secs_f32()
    );

    let start_time = Instant::now();
    //let threshold = elevation_threshold(&points, &tree, k);
    let threshold = 0.1;
    let end_time = Instant::now();
    println!(
        "Computed threshold {} ({} s)",
        threshold,
        (end_time - start_time).as_secs_f32()
    );

    let start_time = Instant::now();
    propagate_ground(&mut points, &tree, &mut ground_point_indices, threshold, k);
    let end_time = Instant::now();

    println!(
        "Created ground! ({} s)",
        (end_time - start_time).as_secs_f32()
    );

    for i in ground_point_indices {
        points[i].is_ground = true;
    }

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

    write_ply(&points, args.output).unwrap();
}
