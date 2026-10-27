//! `rdo_distortion_scale / 256` dumped from the reference build.
//! High halves of `src/tables_rdo.inc` (Linux gcc, bit-exact with MSVC).
//! 4:4:4 is level-major, component, band 0..3. 4:2:0 lists only bands the
//! encoder transmits (level 4 includes band 0; lower levels start at band 1;
//! level 0 has no chroma).

pub const SCALE_444: [f32; 60] = [
    f32::from_bits(0x454e51b3),
    f32::from_bits(0x42acf09c),
    f32::from_bits(0x42acf09c),
    f32::from_bits(0x404b7dab),
    f32::from_bits(0x444e51b3),
    f32::from_bits(0x41acf09c),
    f32::from_bits(0x41acf09c),
    f32::from_bits(0x3f4b7dab),
    f32::from_bits(0x444e51b3),
    f32::from_bits(0x41acf09c),
    f32::from_bits(0x41acf09c),
    f32::from_bits(0x3f4b7dab),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x45349f5f),
    f32::from_bits(0x45349f5f),
    f32::from_bits(0x43c91e83),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x44349f5f),
    f32::from_bits(0x44349f5f),
    f32::from_bits(0x42c91e83),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x44349f5f),
    f32::from_bits(0x44349f5f),
    f32::from_bits(0x42c91e83),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457402c3),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x447402c3),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x447402c3),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457640f8),
    f32::from_bits(0x497640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
];

pub const SCALE_420: [f32; 42] = [
    f32::from_bits(0x42acf09c),
    f32::from_bits(0x42acf09c),
    f32::from_bits(0x404b7dab),
    f32::from_bits(0x45349f5f),
    f32::from_bits(0x45349f5f),
    f32::from_bits(0x43c91e83),
    f32::from_bits(0x43820c59),
    f32::from_bits(0x43820c59),
    f32::from_bits(0x4210ce4b),
    f32::from_bits(0x43820c59),
    f32::from_bits(0x43820c59),
    f32::from_bits(0x4210ce4b),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x457402c3),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x43afb012),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x43afb012),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x45b14d80),
    f32::from_bits(0x45b14d80),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x45b14d80),
    f32::from_bits(0x45b14d80),
    f32::from_bits(0x44b14d80),
    f32::from_bits(0x497640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
    f32::from_bits(0x487640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x477640f8),
    f32::from_bits(0x467640f8),
];

pub fn distortion_scale(chroma_420: bool, level: usize, component: usize, band: usize) -> f32 {
    if chroma_420 {
        let mut i = 0usize;
        for lv in 0..5 {
            for comp in 0..3 {
                if lv == 0 && comp != 0 {
                    continue;
                }
                let band0 = if lv == 4 { 0 } else { 1 };
                for b in band0..4 {
                    if lv == level && comp == component && b == band {
                        return SCALE_420[i];
                    }
                    i += 1;
                }
            }
        }
    } else {
        return SCALE_444[(level * 3 + component) * 4 + band];
    }
    1.0
}
