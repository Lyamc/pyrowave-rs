use pyrowave_rs::{
    parse_sequence_header, payload_words_of, write_packet_list, Backend, Chroma, CreateInfo, Encoder,
    YuvFrame,
};

fn planes(chroma: Chroma) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (cw, ch) = match chroma {
        Chroma::Yuv420 => (32usize, 32usize),
        Chroma::Yuv444 => (64, 64),
    };
    let mut y = vec![0u8; 64 * 64];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for i in 0..y.len() {
        y[i] = ((i * 3 + (i / 64) * 5) & 255) as u8;
    }
    for i in 0..u.len() {
        u[i] = (96 + (i & 31)) as u8;
        v[i] = (160 + (i * 2 & 31)) as u8;
    }
    (y, u, v)
}

fn encode_once(chroma: Chroma) -> (Vec<u8>, pyrowave_rs::PacketList) {
    let (y, u, v) = planes(chroma);
    let mut enc = Encoder::create(CreateInfo {
        backend: Backend::Cpu,
        allow_software: false,
    })
    .expect("cpu encoder");
    assert!(!enc.uses_gpu());
    let frame = YuvFrame {
        width: 64,
        height: 64,
        chroma,
        y: &y,
        u: &u,
        v: &v,
    };
    enc.encode(&frame).expect("encode");
    let boundary = 1 << 20;
    let n = enc.compute_num_packets(boundary).expect("count");
    assert!(n >= 1);
    let mut bits = vec![0u8; boundary];
    let packets = enc.packetize(boundary, &mut bits).expect("packetize");
    assert_eq!(packets.len(), n);
    let len: usize = packets.iter().map(|p| p.size).sum();
    assert!(len > 0);
    bits.truncate(len);
    (bits, packets)
}

fn check(chroma: Chroma) {
    let (a, pa) = encode_once(chroma);
    let (b, pb) = encode_once(chroma);
    assert_eq!(a, b, "two encodes of the same frame differ");
    assert_eq!(pa, pb);
    let (w, h, chroma_444) = parse_sequence_header(&a).expect("sequence header");
    assert_eq!((w, h), (64, 64));
    assert_eq!(chroma_444, chroma == Chroma::Yuv444);
    let sum: usize = pa.iter().map(|p| p.size).sum();
    assert_eq!(sum, a.len());
    let list = write_packet_list(&a, &pa).expect("packet list");
    let count = u32::from_le_bytes(list[0..4].try_into().unwrap());
    assert_eq!(count as usize, pa.len());
    // First packet starts with the sequence header, then a block. The reference
    // payload_words field is the whole packet in words, including its 8-byte header.
    let first = &a[pa[0].offset..pa[0].offset + pa[0].size];
    assert!(first.len() > 16, "packet holds a sequence header and a block");
    let block = &first[8..];
    let words = payload_words_of(block).expect("block header");
    assert!(words >= 2, "block packet includes its header");
    assert!(words * 4 <= block.len(), "payload_words covers the first block");
    eprintln!(
        "cpu {chroma:?} bytes={} packets={} header={w}x{h} payload_words={words} identical=true",
        a.len(),
        pa.len()
    );
}

#[test]
fn cpu_420_and_444_round_trip_the_record() {
    check(Chroma::Yuv420);
    check(Chroma::Yuv444);
}
