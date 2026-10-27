use pyrowave_rs::{
    band_texture_desc, plane_texture_desc, shader_layout_matches_ports, submission_plan,
    DWT_PUSH_BYTES, FORMAT_R32_FLOAT, FORMAT_R8_UNORM, PACK_JOB_BYTES, QUANT_PUSH_BYTES,
    SRV_DIM_TEXTURE2DARRAY, UAV_DIM_TEXTURE2DARRAY,
};

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// The plan `d3d11::encode` submits: 4:2:0 is five batches (13 slots), each
/// push is the 192-byte Texture2DArray cbuffer, quant is the 32x32 grid, and
/// pack jobs carry sequence 1.
#[test]
fn d3d11_encode_submits_the_cs5_pyramid() {
    assert!(shader_layout_matches_ports());

    let plan = submission_plan(64, 64, false).expect("420 plan");
    assert_eq!(plan.dwt.len(), 5, "420 is five dispatches, not one");
    let slots: u32 = plan.dwt.iter().map(|b| b.groups_z).sum();
    assert_eq!(slots, 13);
    assert_eq!(plan.dwt[0].groups_z, 1);
    assert_eq!(plan.dwt[0].slots.len(), 1);
    assert!(plan.dwt[0].slots[0].from_plane);
    assert_eq!(plan.dwt[0].slots[0].level, 0);
    assert_eq!(plan.dwt[1].groups_z, 3);
    assert!(!plan.dwt[1].slots[0].from_plane);
    assert!(plan.dwt[1].slots[1].from_plane);
    assert!(plan.dwt[1].slots[2].from_plane);

    let push = &plan.dwt[0].push;
    assert_eq!(push.len(), DWT_PUSH_BYTES);
    assert_eq!(le_u32(push, 0), 64);
    assert_eq!(le_u32(push, 4), 64);
    assert_eq!(le_u32(push, 16), 128);
    assert_ne!(le_u32(push, 56), 0, "store_mask must be written");
    assert!(plan.dwt.iter().all(|b| b.push.len() == DWT_PUSH_BYTES));

    assert!(!plan.quant.is_empty());
    let coarse = &plan.quant[0];
    assert_eq!((coarse.level, coarse.component), (4, 0));
    assert_eq!(coarse.push.len(), QUANT_PUSH_BYTES);
    assert_eq!(le_u32(&coarse.push, 0), 4);
    assert_eq!(le_u32(&coarse.push, 28), 4);
    assert_eq!(coarse.groups_x, 1);
    assert_eq!(coarse.groups_z, 4);

    let luma0 = plan
        .quant
        .iter()
        .find(|q| q.level == 0 && q.component == 0)
        .expect("level-0 luma quant");
    assert_eq!(le_u32(&luma0.push, 0), 64);
    // quant.hlsl covers 32x32 per group. (64+7)/8 would be the OpenGL grid.
    assert_eq!(luma0.groups_x, 2);
    assert_eq!(luma0.groups_y, 2);
    assert!(plan.quant.iter().all(|q| q.push.len() == QUANT_PUSH_BYTES));

    assert_eq!(plan.pack_jobs.len() % PACK_JOB_BYTES, 0);
    assert!(plan.pack_jobs.len() >= PACK_JOB_BYTES);
    assert_eq!(le_u32(&plan.pack_jobs, 28), 1);
    assert!(plan.total32 > 0);
    assert_eq!(plan.pack_groups_z as usize, plan.pack_jobs.len() / PACK_JOB_BYTES);

    let plane = plane_texture_desc(64, 64);
    assert_eq!(plane.format, FORMAT_R8_UNORM);
    assert_eq!(plane.array_size, 1);
    assert_eq!(plane.srv_dimension, SRV_DIM_TEXTURE2DARRAY);
    let band = band_texture_desc(64, 64);
    assert_eq!(band.format, FORMAT_R32_FLOAT);
    assert_eq!(band.array_size, 4);
    assert_eq!(band.srv_dimension, SRV_DIM_TEXTURE2DARRAY);
    assert_eq!(band.uav_dimension, UAV_DIM_TEXTURE2DARRAY);
    assert_eq!(band.srv_array_size, 4);
    assert_eq!(band.uav_array_size, 4);

    let hd = submission_plan(1280, 720, false).expect("720 plan");
    assert_eq!(hd.dwt.len(), 5, "720p still runs every DWT level");
    assert!(hd.total32 > 0 && hd.total32 <= 60, "720p quant/pack stays inside the 60-block prefix");
    let fhd = submission_plan(1920, 1080, false).expect("1080 plan");
    assert_eq!(fhd.dwt.len(), 5, "1080p still runs every DWT level");
    assert!(fhd.total32 > 0 && fhd.total32 <= 96, "1080p quant/pack stays inside the 96-block prefix");

    let plan444 = submission_plan(64, 64, true).expect("444 plan");
    assert_eq!(plan444.dwt.len(), 15);
    assert!(plan444.dwt.iter().all(|b| b.groups_z == 1 && b.push.len() == DWT_PUSH_BYTES));
    assert!(plan444.dwt[..3].iter().all(|b| b.slots[0].from_plane && b.slots[0].level == 0));
    assert_ne!(le_u32(&plan444.dwt[0].push, 8), 0);
}
