pub fn smallest_eigenvector(m: [[f64; 3]; 3]) -> [f64; 3] {
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
    if length2(v) < 1e-9 {
        v = cross(a[0], a[2]);
    }
    if length2(v) < 1e-9 {
        v = cross(a[1], a[2]);
    }
    normalize(v)
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn length2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

pub fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = length2(v).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0] // degenerate fallback
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}
