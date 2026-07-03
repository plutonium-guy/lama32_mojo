"""Multi-shard safetensors reader for streaming inference.

Parses model.safetensors.index.json (tensor -> shard filename), then each
shard's header for byte offsets. get(name) yields the shard path + absolute
byte range so the decoder can seek+read one tensor at a time (weights never
all resident). Single-file models work too (no index -> the one shard).
"""

from std.python import Python, PythonObject
from std.collections import Dict


def _pyint(o: PythonObject) raises -> Int:
    return atol(String(o))


@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var name: String
    var dtype: String
    var shape: List[Int]
    var path: String            # which shard file this tensor lives in
    var begin: Int              # absolute byte offset in that shard
    var end: Int

    def numel(self) -> Int:
        var p = 1
        for i in range(len(self.shape)):
            p *= self.shape[i]
        return p

    def nbytes(self) -> Int:
        return self.end - self.begin


struct SafeTensors(Movable):
    var infos: List[TensorInfo]
    var index: Dict[String, Int]

    def __init__(out self, model_dir: String) raises:
        self.infos = List[TensorInfo]()
        self.index = Dict[String, Int]()
        var os = Python.import_module("os")
        var js = Python.import_module("json")
        var st = Python.import_module("struct")
        var bi = Python.import_module("builtins")

        # collect shard files (via index.json if present, else glob one file)
        var shards = List[String]()
        var idx_path = String(os.path.join(model_dir, "model.safetensors.index.json"))
        if os.path.exists(idx_path):
            var idx = js.loads(bi.open(idx_path, "r").read())
            var wm = idx["weight_map"]
            var seen = Dict[String, Int]()
            var keys = bi.list(wm.keys())
            for i in range(len(keys)):
                var fname = String(wm[keys[i]])
                if fname not in seen:
                    seen[fname] = 1
                    shards.append(String(os.path.join(model_dir, fname)))
        else:
            shards.append(String(os.path.join(model_dir, "model.safetensors")))

        for si in range(len(shards)):
            var path = shards[si]
            var f = bi.open(path, "rb")
            var n = _pyint(st.unpack("<Q", f.read(8))[0])
            var hdr = js.loads(f.read(n))
            f.close()
            var data_start = 8 + n
            var hkeys = bi.list(hdr.keys())
            for i in range(len(hkeys)):
                var k = String(hkeys[i])
                if k == "__metadata__":
                    continue
                var m = hdr[k]
                var offs = m["data_offsets"]
                var shp = m["shape"]
                var shape = List[Int]()
                for j in range(len(shp)):
                    shape.append(_pyint(shp[j]))
                self.index[k] = len(self.infos)
                self.infos.append(
                    TensorInfo(k, String(m["dtype"]), shape^, path,
                               data_start + _pyint(offs[0]),
                               data_start + _pyint(offs[1]))
                )

    def num_tensors(self) -> Int:
        return len(self.infos)

    def get(self, name: String) raises -> TensorInfo:
        return self.infos[self.index[name]].copy()

    def has(self, name: String) -> Bool:
        return name in self.index
