//! Little-endian PyroWave record. Layout matches the bitfields in
//! `src/frame.cpp` (`BitstreamSequenceHeader`, `BitstreamHeader`) and the
//! packet walk in `packetize_blocks`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet {
    pub offset: usize,
    pub size: usize,
}

thread_local! {
    static PACKET_POOL: std::cell::RefCell<Vec<Vec<Packet>>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_packet_vec() -> Vec<Packet> {
    PACKET_POOL.with(|pool| pool.borrow_mut().pop().unwrap_or_default())
}

/// Packet list returned by `Encoder::packetize`. Dropping it keeps the allocation.
pub struct PacketList {
    packets: Vec<Packet>,
}

impl Drop for PacketList {
    fn drop(&mut self) {
        let mut packets = std::mem::take(&mut self.packets);
        packets.clear();
        PACKET_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.len() < 4 {
                pool.push(packets);
            }
        });
    }
}

impl std::ops::Deref for PacketList {
    type Target = [Packet];
    fn deref(&self) -> &[Packet] {
        &self.packets
    }
}

impl std::fmt::Debug for PacketList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.packets.fmt(f)
    }
}

impl PartialEq for PacketList {
    fn eq(&self, other: &Self) -> bool {
        self.packets == other.packets
    }
}

impl PacketList {
    pub(crate) fn from_vec(packets: Vec<Packet>) -> Self {
        Self { packets }
    }
}

#[derive(Clone, Debug)]
pub struct BlockMeta {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FrameBits {
    pub width: u32,
    pub height: u32,
    pub chroma_444: bool,
    pub blocks: Vec<BlockMeta>,
}

/// `PWV1Header` for limited-range BT.709 4:2:0, left chroma siting, 8-bit.
/// Sequence, extended, total_blocks, and code stay 0 in the file header.
pub fn pwv1_header(width: u32, height: u32, fps_num: u32, fps_den: u32) -> [u8; 24] {
    let mut word0 = (width - 1) & 0x3fff;
    word0 |= ((height - 1) & 0x3fff) << 14;
    let word1 = (1u32 << 30) | (1u32 << 31);
    let mut out = [0u8; 24];
    out[0..4].copy_from_slice(b"PWV1");
    out[4..8].copy_from_slice(&word0.to_le_bytes());
    out[8..12].copy_from_slice(&word1.to_le_bytes());
    out[12..16].copy_from_slice(&fps_num.to_le_bytes());
    out[16..20].copy_from_slice(&fps_den.to_le_bytes());
    out[20] = 8;
    out[21] = 1;
    out
}

/// One-frame PWV1 file: the 24-byte header, a little-endian payload length, then
/// the bytes `Encoder::packetize` wrote.
pub fn pwv1_one_frame(width: u32, height: u32, fps_num: u32, fps_den: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(28 + payload.len());
    out.extend_from_slice(&pwv1_header(width, height, fps_num, fps_den));
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Sequence header, 8 bytes, LSB-first bitfields.
pub fn sequence_header(width: u32, height: u32, total_blocks: u32, chroma_444: bool) -> [u8; 8] {
    sequence_header_with(width, height, total_blocks, chroma_444, 1)
}

pub fn sequence_header_with(
    width: u32,
    height: u32,
    total_blocks: u32,
    chroma_444: bool,
    sequence: u32,
) -> [u8; 8] {
    let mut word0 = (width - 1) & 0x3fff;
    word0 |= ((height - 1) & 0x3fff) << 14;
    word0 |= (sequence & 7) << 28;
    word0 |= 1u32 << 31; // extended
    let mut word1 = total_blocks & 0x00ff_ffff;
    if chroma_444 {
        word1 |= 1 << 26;
    }
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&word0.to_le_bytes());
    out[4..].copy_from_slice(&word1.to_le_bytes());
    out
}

/// One block: 8-byte header plus `payload` extra bytes (multiple of 4).
pub fn block_record(block_index: u32, payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() % 4 == 0, "payload is whole words");
    let payload_words = (payload.len() / 4) as u16;
    let second: u16 = (payload_words & 0x0fff) | (1 << 12) | (1 << 15);
    let third: u32 = block_index << 8; // quant_code = 0
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&0u16.to_le_bytes()); // ballot
    out.extend_from_slice(&second.to_le_bytes());
    out.extend_from_slice(&third.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn payload_words_of(block: &[u8]) -> Option<usize> {
    if block.len() < 4 {
        return None;
    }
    let second = u16::from_le_bytes([block[2], block[3]]);
    Some((second & 0x0fff) as usize)
}

/// Packet count `packetize` would return, without building the packet list.
pub fn count_packets(frame: &FrameBits, packet_boundary: usize) -> Result<usize, String> {
    if frame.blocks.is_empty() {
        return Err("no blocks".into());
    }
    if packet_boundary < 8 {
        return Err("packet boundary is smaller than the sequence header".into());
    }
    let mut packets = 0usize;
    let mut size_in_packet = 8usize;
    for block in &frame.blocks {
        let nbytes = block.bytes.len();
        if nbytes == 0 {
            continue;
        }
        if size_in_packet + nbytes > packet_boundary {
            packets += 1;
            size_in_packet = 0;
        }
        size_in_packet += nbytes;
    }
    if size_in_packet > 0 {
        packets += 1;
    }
    if packets == 0 {
        return Err("packetize produced no packets".into());
    }
    Ok(packets)
}

/// Walk blocks the way `packetize_blocks` does. Returns the packet count, or
/// an error when a block does not fit in `output`.
pub fn packetize(
    frame: &FrameBits,
    packet_boundary: usize,
    output: Option<&mut [u8]>,
) -> Result<(Vec<Packet>, usize), String> {
    if frame.blocks.is_empty() {
        return Err("no blocks".into());
    }
    if packet_boundary < 8 {
        return Err("packet boundary is smaller than the sequence header".into());
    }
    let nonzero = frame.blocks.iter().filter(|b| !b.bytes.is_empty()).count() as u32;
    let sequence = frame
        .blocks
        .iter()
        .find_map(|b| {
            if b.bytes.len() >= 4 {
                let word = u32::from_le_bytes(b.bytes[0..4].try_into().unwrap());
                Some((word >> 28) & 7)
            } else {
                None
            }
        })
        .unwrap_or(1);
    let header = sequence_header_with(frame.width, frame.height, nonzero, frame.chroma_444, sequence);

    let mut packets = take_packet_vec();
    packets.clear();
    let mut size_in_packet = 0usize;
    let mut packet_offset = 0usize;
    let mut output_offset = 0usize;

    let write = |dst: &mut Option<&mut [u8]>, at: usize, src: &[u8]| -> Result<(), String> {
        if let Some(buf) = dst.as_deref_mut() {
            let end = at + src.len();
            if end > buf.len() {
                return Err("bitstream buffer is smaller than the record".into());
            }
            buf[at..end].copy_from_slice(src);
        }
        Ok(())
    };

    let mut out = output;
    write(&mut out, 0, &header)?;
    output_offset += header.len();
    size_in_packet += header.len();

    for block in &frame.blocks {
        let nbytes = block.bytes.len();
        if nbytes == 0 {
            continue;
        }
        if size_in_packet + nbytes > packet_boundary {
            packets.push(Packet {
                offset: packet_offset,
                size: size_in_packet,
            });
            size_in_packet = 0;
            packet_offset = output_offset;
        }
        write(&mut out, output_offset, &block.bytes)?;
        output_offset += nbytes;
        size_in_packet += nbytes;
    }
    if size_in_packet > 0 {
        packets.push(Packet {
            offset: packet_offset,
            size: size_in_packet,
        });
    }
    if packets.is_empty() {
        return Err("packetize produced no packets".into());
    }
    Ok((packets, output_offset))
}

/// Outer list the C callers write after packetize: little-endian count, then
/// each packet's little-endian size and its bytes.
pub fn write_packet_list(bitstream: &[u8], packets: &[Packet]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(&(packets.len() as u32).to_le_bytes());
    for packet in packets {
        let end = packet.offset.checked_add(packet.size).ok_or("packet range")?;
        if end > bitstream.len() {
            return Err("packet extends past the bitstream".into());
        }
        out.extend_from_slice(&(packet.size as u32).to_le_bytes());
        out.extend_from_slice(&bitstream[packet.offset..end]);
    }
    Ok(out)
}

/// GPU pack shaders write `BitstreamPacket` pairs (`offset_u32`, `num_words`)
/// and the block bytes at `offset_u32 * 4`. Empty blocks are skipped, matching
/// `packetize_blocks` in the C encoders. The kept records are concatenated into
/// one block so packetize copies the payload once. A record past the read-back
/// prefix is dropped, matching `frame.cpp` when the copy is shorter than the
/// word counter.
pub fn blocks_from_gpu(bitstream: &[u8], meta: &[u8]) -> Result<Vec<BlockMeta>, String> {
    if meta.len() % 8 != 0 {
        return Err("GPU block meta length is not a multiple of 8".into());
    }
    let mut spans = Vec::new();
    let mut total = 0usize;
    for rec in meta.chunks_exact(8) {
        let offset_u32 = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        let num_words = u32::from_le_bytes(rec[4..8].try_into().unwrap());
        if num_words == 0 {
            continue;
        }
        let start = (offset_u32 as usize).saturating_mul(4);
        let nbytes = (num_words as usize).saturating_mul(4);
        let end = match start.checked_add(nbytes) {
            Some(end) => end,
            None => continue,
        };
        if end > bitstream.len() {
            continue;
        }
        spans.push((start, end));
        total += nbytes;
    }
    if spans.is_empty() {
        return Err("GPU pack wrote no block packets".into());
    }
    let mut bytes = Vec::with_capacity(total);
    for (start, end) in spans {
        bytes.extend_from_slice(&bitstream[start..end]);
    }
    Ok(vec![BlockMeta { bytes }])
}

#[cfg(test)]
mod tests {
    use super::{blocks_from_gpu, count_packets, packetize, BlockMeta, FrameBits};

    #[test]
    fn count_packets_matches_packetize_splits() {
        let frame = FrameBits {
            width: 1280,
            height: 720,
            chroma_444: false,
            blocks: vec![
                BlockMeta { bytes: vec![1, 2, 3, 4] },
                BlockMeta { bytes: Vec::new() },
                BlockMeta { bytes: vec![5, 6, 7, 8, 9, 10, 11, 12] },
            ],
        };
        let mut out = vec![0u8; 64];
        let (packets, _) = packetize(&frame, 12, Some(&mut out)).expect("packetize");
        let n = count_packets(&frame, 12).expect("count");
        assert_eq!(n, packets.len());
        assert!(n > 1);
    }

    #[test]
    fn blocks_from_gpu_joins_live_records_and_skips_holes() {
        let mut bitstream = vec![0u8; 32];
        bitstream[0..4].copy_from_slice(&1u32.to_le_bytes());
        bitstream[16..20].copy_from_slice(&2u32.to_le_bytes());
        let mut meta = Vec::new();
        meta.extend_from_slice(&0u32.to_le_bytes());
        meta.extend_from_slice(&1u32.to_le_bytes());
        meta.extend_from_slice(&0u32.to_le_bytes());
        meta.extend_from_slice(&0u32.to_le_bytes());
        meta.extend_from_slice(&4u32.to_le_bytes());
        meta.extend_from_slice(&1u32.to_le_bytes());
        meta.extend_from_slice(&100u32.to_le_bytes());
        meta.extend_from_slice(&4u32.to_le_bytes());
        let blocks = blocks_from_gpu(&bitstream, &meta).expect("records");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].bytes, [1, 0, 0, 0, 2, 0, 0, 0]);
    }
}

pub fn parse_sequence_header(bytes: &[u8]) -> Option<(u32, u32, bool)> {
    if bytes.len() < 8 {
        return None;
    }
    let word0 = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
    let word1 = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let width = (word0 & 0x3fff) + 1;
    let height = ((word0 >> 14) & 0x3fff) + 1;
    let chroma_444 = (word1 >> 26) & 1 == 1;
    Some((width, height, chroma_444))
}
