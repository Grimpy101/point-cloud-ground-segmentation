use std::path::Path;

use itertools::izip;
use las::point::Classification;

use crate::point::Point;

pub fn write_laz(
    points: &[Point],
    transforms: las::Vector<las::Transform>,
    path: &Path,
) -> std::io::Result<()> {
    let mut format = las::point::Format::new(6).unwrap();
    format.extra_bytes = 1;

    let mut builder = las::Builder::from((1, 4));
    builder.point_format = format;
    builder.transforms = transforms;

    let header = builder.into_header().unwrap();
    let mut writer = las::Writer::from_path(path, header).unwrap();

    for p in points {
        let las_point = las::Point {
            x: p.x,
            y: p.y,
            z: p.z,
            classification: if p.is_ground {
                Classification::Ground
            } else {
                Classification::Unclassified
            },
            gps_time: Some(0.0),
            extra_bytes: vec![p.wrong],
            ..Default::default()
        };
        writer.write_point(las_point).unwrap();
    }
    writer.close().unwrap();
    Ok(())
}

pub fn read_las(path: &Path, test: bool) -> (Vec<Point>, Vec<u8>, las::Vector<las::Transform>) {
    let mut reader = las::Reader::from_path(path).unwrap();
    let n = reader.header().number_of_points() as usize;
    let mut points = Vec::with_capacity(n);
    let mut classes = if test {
        Vec::with_capacity(n)
    } else {
        Vec::new()
    };

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
        if test {
            classes.push(cls);
        }
    }

    (points, classes, *reader.header().transforms())
}
