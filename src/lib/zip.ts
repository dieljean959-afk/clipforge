// Minimal ZIP writer (stored entries, no compression) so "Export all 9:16"
// works fully client-side without a dependency. Supports CRC-32 over the
// bundled file payloads.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

interface ZipEntry {
  name: string;
  data: Uint8Array;
}

function dosDateTime(date: Date): { time: number; date: number } {
  const time =
    (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const d =
    ((date.getFullYear() - 1980) << 9) |
    ((date.getMonth() + 1) << 5) |
    date.getDate();
  return { time, date: d };
}

/**
 * Build a ZIP archive from text entries. Returns a Blob ready for download.
 */
export function buildZip(
  files: { name: string; text: string }[],
  opts: { date?: Date } = {},
): Blob {
  const when = dosDateTime(opts.date ?? new Date());
  const enc = new TextEncoder();
  const toPart = (u8: Uint8Array): BlobPart =>
    u8.buffer instanceof ArrayBuffer && u8.byteOffset === 0
      ? new Uint8Array(u8.buffer.slice(0) as ArrayBuffer)
      : (u8.slice().buffer as ArrayBuffer);
  const entries: ZipEntry[] = files.map((f) => ({
    name: f.name,
    data: enc.encode(f.text),
  }));

  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  const u16 = (v: number) => new Uint8Array([v & 0xff, (v >>> 8) & 0xff]);
  const u32 = (v: number) =>
    new Uint8Array([v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff]);

  for (const entry of entries) {
    const nameBytes = enc.encode(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    const local = new Uint8Array(30 + nameBytes.length);
    local.set(u32(0x04034b50), 0); // local file header signature
    local.set(u16(20), 4); // version needed
    local.set(u16(0), 6); // flags
    local.set(u16(0), 8); // method: store
    local.set(u16(when.time), 10);
    local.set(u16(when.date), 12);
    local.set(u32(crc), 14);
    local.set(u32(size), 18); // compressed
    local.set(u32(size), 22); // uncompressed
    local.set(u16(nameBytes.length), 26);
    local.set(u16(0), 28);
    local.set(nameBytes, 30);

    chunks.push(local, entry.data);

    const cd = new Uint8Array(46 + nameBytes.length);
    cd.set(u32(0x02014b50), 0); // central directory signature
    cd.set(u16(20), 4); // version made by
    cd.set(u16(20), 6); // version needed
    cd.set(u16(0), 8);
    cd.set(u16(0), 10);
    cd.set(u16(when.time), 12);
    cd.set(u16(when.date), 14);
    cd.set(u32(crc), 16);
    cd.set(u32(size), 20);
    cd.set(u32(size), 24);
    cd.set(u16(nameBytes.length), 28);
    cd.set(u32(offset), 42); // relative offset of local header
    cd.set(nameBytes, 46);
    central.push(cd);

    offset += local.length + size;
  }

  const centralSize = central.reduce((acc, c) => acc + c.length, 0);
  const end = new Uint8Array(22);
  end.set(u32(0x06054b50), 0);
  end.set(u16(entries.length), 8);
  end.set(u16(entries.length), 10);
  end.set(u32(centralSize), 12);
  end.set(u32(offset), 16);

  const parts: BlobPart[] = [...chunks, ...central, end].map(toPart);
  return new Blob(parts, { type: "application/zip" });
}

/** Trigger a client-side download of a blob under the given filename. */
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function downloadText(text: string, filename: string) {
  downloadBlob(new Blob([text], { type: "text/plain;charset=utf-8" }), filename);
}
