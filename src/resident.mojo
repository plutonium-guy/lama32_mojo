"""Resident bf16 weight arena for small models (llama32, qwen3).

Uploads all tensors under one GPU map_to_host() — avoids hundreds of
per-tensor maps during load (~5-10x faster startup).
"""

from std.time import sleep
from std.gpu.host import DeviceContext, DeviceBuffer
from std.collections import Dict
from std.memory import memcpy
from safetensors import SafeTensors
from llama_common import PAD


struct Weights(Movable):
    var buf: DeviceBuffer[DType.uint16]
    var offs: Dict[String, Int]
    var top: Int
    var cap: Int

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        self.buf = ctx.enqueue_create_buffer[DType.uint16](cap)
        self.offs = Dict[String, Int]()
        self.top = PAD
        self.cap = cap

    def o(self, name: String) raises -> Int:
        return self.offs[name]

    def upload_all(mut self, st: SafeTensors, names: List[String]) raises:
        """Disk -> GPU arena in a single host map (one map for all tensors)."""
        with self.buf.map_to_host() as h:
            var base = h.unsafe_ptr()
            for i in range(len(names)):
                var name = names[i]
                var info = st.get(name)
                var count = info.numel()
                if self.top + count > self.cap:
                    raise Error("weight arena overflow at " + name)
                var raw: List[UInt8]
                var attempt = 0
                while True:
                    try:
                        var f = open(info.path, "r")
                        _ = f.seek(UInt64(info.begin))
                        raw = f.read_bytes(count * 2)
                        f.close()
                        break
                    except e:
                        attempt += 1
                        if attempt >= 15:
                            raise e.copy()
                        print("  read failed (", e, "), retry", attempt)
                        sleep(2.0)
                memcpy(dest=(base + self.top).bitcast[UInt8](),
                       src=raw.unsafe_ptr(), count=count * 2)
                self.offs[name] = self.top
                self.top += count
                if i % 40 == 0:
                    print("  ", i, "/", len(names))
