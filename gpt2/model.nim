import math
import arraymancer
import sequtils
import options
import strformat
import strutils
import tables
import uri
import os


type
  HParams* = object
    n_vocab: int
    n_ctx: int
    n_embd: int
    n_head: int
    n_layer: int
  HParamsRef* = ref HParams

  GPT2ModelRef* = ref object
    current_scope: string
    model_path: string
    hparams*: HParams
    cache: TableRef[string, Tensor[float32]]

func defaultHParams(): HParams =
  result.n_vocab = 50257
  result.n_ctx = 1024
  result.n_embd = 768
  result.n_head = 12
  result.n_layer = 12

func newGPT2Model(model_path: string, hparams: HParams): GPT2ModelRef =
  new result
  result.current_scope = "model"
  result.model_path = model_path
  result.hparams = hparams
  result.cache = newTable[string, Tensor[float32]]()

func scope(self: GPT2ModelRef, scope: string): GPT2ModelRef =
  new result
  result.current_scope = self.current_scope & "/" & scope
  result.model_path = self.model_path
  result.hparams = self.hparams
  result.cache = self.cache

proc get_tensor[T](self: GPT2ModelRef, name: string): Tensor[T] =
  let filename = (self.current_scope & "/" & name).replace("/", "_") & ":0.npy"
  if self.cache.contains(filename):
    return self.cache[filename]
  #echo "loading:", filename
  result = read_npy[T](joinPath(self.model_path, filename))
  self.cache[filename] = result

proc get_tensor(self: GPT2ModelRef, name: string, T: typedesc): Tensor[T] =
  get_tensor[T](self, name)

template `of`(axis: int, x: Tensor): int =
  if axis < 0:
    x.shape.len + axis
  else:
    axis

func fix_shape(x: Tensor, shape: openarray[int]): seq[int] =
  let x_product = x.shape.product
  var shape_product = 1
  result = @[]
  for n in shape:
    if n != -1:
      # TODO: ensure there is only one -1
      shape_product *= n
  for n in shape:
    if n == -1:
      result.add(x_product div shape_product)
    else:
      result.add(n)

func range_tensor(l, h: int): Tensor[int] =
  (l ..< h).toSeq().toTensor()

func range_tensor(h: int): Tensor[int] =
  range_tensor(0, h)

func softmax(x: Tensor, axis: int = -1): Tensor =
  let ex = exp(x .- max(x, axis=axis of x))
  ex ./ sum(ex, axis=axis of x)

func rsqrt[F: SomeNumber](x: F): F =
  F(1.0 / sqrt(float x))

func rsqrt[F](x: Tensor[F]): Tensor[F] =
  F(1.0) ./ sqrt(x)

func gelu[F](x: Tensor[F]): Tensor[F] =
  F(0.5) * x .* (F(1.0) .+ tanh((x + (F(0.044715) * (x .^ 3))) * F(sqrt(2.0 / math.PI))))

func reduce_mean(x: Tensor, axis: int, keepdims: bool): Tensor =
  result = mean(x, axis of x)
  if not keepdims:
    result = result.squeeze(axis of x)

func norm[T](x, g, b: Tensor[T], axis: int = -1, epsilon: float=1e-5): Tensor[T] =
  # Normalize to mean = 0, std = 1, then do a diagonal affine transform.
  let u = reduce_mean(x, axis=axis, keepdims=true)
  let s = reduce_mean(square(x .- u), axis=axis, keepdims=true)
  let x1 = (x .- u) .* rsqrt(s .+ T(epsilon))
  (x1 .* g) .+ b

func split_states(x: Tensor, n: int): Tensor =
  # Reshape the last dimension of x into [n, x.shape[-1]/n].

  let start = x.shape[0 ..< ^1].toSeq()
  let m = x.shape[^1]
  let shape = start.concat(@[n, m div n])
  reshape(x, shape)

func merge_states(x: Tensor): Tensor =
  # Smash the last two dimensions of x into a single dimension.
  let start = x.shape[0 ..< ^2].toSeq()
  let a = x.shape[^2]
  let b = x.shape[^1]
  let shape = start.concat(@[a*b])
  reshape(x, shape)

proc conv1d(x, w, b: Tensor): Tensor =
  let nf = b.shape[^1]
  let start = x.shape[0 ..< ^1].toSeq()
  let nx = x.shape[^1]
  let shape = start.concat(@[nf])
  let t0 = (reshape(x, fix_shape(x, [-1, nx])) * reshape(w, fix_shape(w, [-1, nf])))
  reshape(t0 .+ b, shape)

func attention_mask[T](nd, ns: int): Tensor[T] =
    #[1's in the lower triangle, counting from the lower right corner.
      Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd)
    ]#
    let i = range_tensor(nd).reshape(nd, 1).broadcast([nd, ns])
    let j = range_tensor(ns).reshape(1, ns)
    let k = j .- ns .+ nd
    let r = map2_inline(i, k.broadcast([nd, ns])):
      if x >= y:
        1
      else:
        0
    r.astype(T)

func split_heads(x: Tensor, n_head: int): Tensor =
  # From [batch, sequence, features] to [batch, heads, sequence, features]
  split_states(x, n_head).permute(0, 2, 1, 3)

func merge_heads(x: Tensor): Tensor =
  # Reverse of split_heads
  return merge_states(x.permute(0, 2, 1, 3))

func mask_attn_weights[T](w: Tensor[T]): Tensor[T] =
  # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
  let nd = w.shape[^2]
  let ns = w.shape[^1]
  let b = attention_mask[T](nd, ns).reshape([1, 1, nd, ns])
  w .* b .- (T(1e10) * (T(1) .- b))

func nth_or_first(x: Tensor, n: int): Tensor =
  if x.shape[0] == 1:
    return x[0, _].squeeze(0)
  else:
    return x[n, _].squeeze(0)

proc matmul[T](a, b: Tensor[T]): Tensor[T] =
  if a.shape.len == 2 and b.shape.len <= 2:
    return a * b

  let start_a = a.shape[0 ..< ^2]
  let start_b = b.shape[0 ..< ^2]

  assert start_a.len == 0 or start_b.len == 0 or start_a == start_b

  let flat_a = a.reshape(a.fix_shape([-1, a.shape[^2], a.shape[^1]]))
  let flat_b = b.reshape(b.fix_shape([-1, b.shape[^2], b.shape[^1]]))

  var result_seq = newSeq[Tensor[T]]()
  for i in 0 ..< max(flat_a.shape[0], flat_b.shape[0]):
    let ea = flat_a.nth_or_first(i)
    let eb = flat_b.nth_or_first(i)
    result_seq.add(ea * eb)

  let inner_shape = @[a.shape[^2], b.shape[^1]]
  let start_shape = if start_a.len == 0:
    start_b
  else:
    start_a

  result_seq.stack().reshape(start_shape.toSeq.concat(inner_shape))

proc multihead_attn[T](q, k, v: Tensor[T]): Tensor[T] =
  # q, k, v have shape [batch, heads, sequence, features]
  var w = matmul(q, k.permute(0, 1, 3, 2))
  w = w * rsqrt(T(v.shape[^1]))
  w = mask_attn_weights(w)
  w = softmax(w, -1)
  matmul(w, v)

proc conv1d[T](self: GPT2ModelRef, x: Tensor[T]): Tensor[T] =
  let w = self.get_tensor("w", T)
  let b = self.get_tensor("b", T)
  return conv1d(x, w, b.reshape(1, b.shape[^1]))

proc attn(self: GPT2ModelRef, x: Tensor, n_state: int, past: Option[Tensor]): tuple[a:Tensor, p:Tensor] =
  assert x.shape.len == 3  # Should be [batch, sequence, features]
  assert n_state mod self.hparams.n_head == 0
  if past.isSome:
      assert past.get.shape.len == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
  let c = self.scope("c_attn").conv1d(x)
  let chunk = c.shape[2] div 3
  let qkv = c.split(chunk, axis=2).map proc(x: auto): auto =
    x.split_heads(self.hparams.n_head)

  let q = qkv[0]
  var k = qkv[1]
  var v = qkv[2]
  let present = stack([k, v], axis=1)
  if past.isSome:
    let pkv = past.get.split(1, axis=1)
    let pk = pkv[0]
    let pv = pkv[1]
    k = concat([pk, k], axis= -2 of k)
    v = concat([pv, v], axis= -2 of v)
  var a = multihead_attn(q, k, v)
  a = merge_heads(a)
  a = self.scope("c_proj").conv1d(a)
  (a, present)

proc mlp(self: GPT2ModelRef, x: Tensor): Tensor =
  let h = self.scope("c_fc").conv1d(x).gelu()
  self.scope("c_proj").conv1d(h)

proc norm[T](self: GPT2ModelRef, x: Tensor[T]): Tensor[T] =
  let g = self.get_tensor("g", T)
  let b = self.get_tensor("b", T)
  norm(x, g.reshape(1, 1, g.shape[^1]), b.reshape(1, 1, b.shape[^1]))

proc layer[T](self: GPT2ModelRef, x: Tensor[T], past: Option[Tensor[T]]): tuple[a:Tensor[T], p:Tensor[T]] =
  let nx = x.shape[^1]
  let norm_x = self.scope("ln_1").norm(x)
  let (a, present) = self.scope("attn").attn(norm_x, nx, past=past)
  var x1 = x + a
  let x2 = self.scope("ln_2").norm(x1)
  let m = self.scope("mlp").mlp(x2)
  x1 = x1 + m
  (x1, present)

func gather[T](x: Tensor[T], indices: Tensor[int]): Tensor[T] =
  var flat_tensors: seq[Tensor[T]] = @[]
  let flat_indices = indices.reshape(indices.fix_shape([-1]))
  for i in 0 ..< flat_indices.shape[0]:
    flat_tensors.add(x.nth_or_first(i))
  let shape = indices.shape.toSeq.concat(x.shape[1 .. ^1].toSeq)
  flat_tensors.stack().reshape(shape)


func expand_tile(value: Tensor, size: int): Tensor =
    var tensors: seq[Tensor] = @[]
    for i in 0 ..< size:
      tensors.add(value)
    return tensors.stack()

proc positions_for(tokens: Tensor, past_length: int): Tensor =
  let batch_size = tokens.shape[0]
  let nsteps = tokens.shape[1]
  expand_tile(past_length .+ range_tensor(nsteps), batch_size)

proc predict[T](self: GPT2ModelRef, x: Tensor[int], past: Option[Tensor[T]]): tuple[logits: Tensor[T], present: Tensor[T]] =
    let batch = x.shape[0]
    let sequence = x.shape[1]

    let wpe = self.get_tensor("wpe", T)
    let wte = self.get_tensor("wte", T)
    let past_length = if past.isNone:
      0
    else:
      past.get.shape[^2]

    var h = gather(wte, x) + gather(wpe, positions_for(x, past_length))

    # Transformer
    var presents: seq[Tensor[T]] = @[]
    for i in 0 ..< self.hparams.n_layer:
      let p: Option[Tensor[T]] =
        if past.isNone:
          none(Tensor[T])
        else:
          some(past.get.nth_or_first(i))
      let r = self.scope(&"h{i}").layer(h, past=p)
      h = r.a
      presents.add(r.p)
    
    h = self.scope("ln_f").norm(h)

    # Language model loss.  Do tokens <n predict token n?
    let h_flat = h.reshape(batch*sequence, self.hparams.n_embd)
    var logits = matmul(h_flat, wte.permute(1, 0))
    logits = logits.reshape(batch, sequence, self.hparams.n_vocab)

    (logits, presents.stack(1))

when isMainModule:
  import logging
  import unittest

  proc runTests*() =
    test "gelu":
      let x = [float32 1.0, 2.0, 3.0].toTensor()
      let y = [float32 0.8411920070648193, 1.95459771156311, 2.996362686157227].toTensor()
      check(gelu(x) == y)

    test "reduce_mean":
      let x = [
        [float32 1.0, 2.0],
        [float32 3.0, 4.0],
      ].toTensor()
      check reduce_mean(x, -1, keepdims=true) == [[float32 1.5], [float32 3.5]].toTensor()
      check reduce_mean(x, 0, keepdims=true) == [[float32 2.0, 3.0]].toTensor()

    test "norm":
      let x = [float32 1.0, 2.0, 3.0].toTensor()
      let g = [float32 2.0, 3.0, 4.0].toTensor()
      let b = [float32 3.0, 2.0, 1.0].toTensor()
      check norm(x, g, b) == [float32 0.5505287647247314,      2.0,     5.898942470550537].toTensor()
    
    test "split_states & merge_states":
      let x = range_tensor(4*3).reshape(3, 4)
      let r = [
        [[ 0,  1],
          [ 2,  3]],
      [[ 4,  5],
        [ 6,  7]],
      [[ 8,  9],
        [10 ,11]]].toTensor()

      check split_states(x, 2) == r
      check merge_states(r) == x

    test "conv1d":
      let x = range_tensor(4)
      let w = [[
          [1,2,3,4],
          [5,6,7,8],
      ]].toTensor()
      let b = [[1, 1]].toTensor()
      check conv1d(x, w, b) == [35, 41].toTensor()
    
    test "fix_shape":
      let x = range_tensor(8)
      check x.fix_shape([-1, 2]) == @[4, 2]
      check x.fix_shape([2, -1, 2]) == @[2, 2, 2]
      check x.fix_shape([4, -1, 2]) == @[4, 1, 2]

    test "attention_mask":
      let mask = [[1, 1, 1, 0], [1, 1, 1, 1]].toTensor()
      check attention_mask[int](2, 4) == mask

    test "split_heads":
      let x = range_tensor(1*4*6).reshape([1, 4, 6])
      let s = split_heads(x, 2)
      check s.shape == [1, 2, 4, 3]

    test "mask_attn_weights":
      # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
      let w = range_tensor(1*2*3*4).reshape([1, 2, 3, 4])
      let mw = mask_attn_weights(w)
      check mw[0, 0, 0, _].reshape([4]) == [0, 1, -int(1e10), -int(1e10)].toTensor()
      check mw[0, 0, 1, _].reshape([4]) == [4, 5, 6, -int(1e10)].toTensor()
    
    test "multihead_attn":
      let (b, h, s, f) = (1, 2, 3, 4)
      let q = range_tensor(b*h*s*f).reshape([b, h, s, f]).astype(float32)
      let k = q
      let v = q
      check multihead_attn(q, k, v).reshape(24) == range_tensor(24).astype(float32)

    test "matmul":
      let a = range_tensor(24).reshape(1, 2, 3, 4)
      let b = a.permute(0, 1, 3, 2)
      check a.matmul(b).shape == [1, 2, 3, 3]

    test "model":
      let self = newGPT2Model("./124Mnp", defaultHParams())
      let x = range_tensor(12).reshape(2, 6)
      let past = none(Tensor[float32])
      let r =  self.predict(x, past)
      check r.logits[0, 0, 0..<10].reshape([10]) == [
        float32 -37.75491333007812, -38.26512145996094, -40.4859504699707, -40.55413055419922, -39.97011947631836,
        -40.66231155395508, -38.56421279907227, -39.10071563720703, -38.16803359985352, -39.44895935058594
        ].toTensor()
      check r.present[0, 0, 0, 0, 0, 0..<10].reshape([10]) == [
        float32 -1.22442102432251, 2.271039247512817, 0.7146419882774353, 0.06725620478391647, 0.484825611114502,
        0.8162559270858765, 0.5466167330741882, 0.7272219061851501, -1.744130253791809, 0.4687796235084534
      ].toTensor()

  addHandler(newConsoleLogger())
  runTests()
  
