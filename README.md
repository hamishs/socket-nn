# socket-nn
Run inference on neural networks over TCP sockets.

Uses `candle` for tensor operations and `tokio` to run the server asynchronously providing easy concurrency. One set of weights is loaded into memory and shared accross concurrent tasks with a Rust's `Arc` type. Tensors are sent and recieved with the `numpy` byte format.

Run an example server with:
```
cargo run --example mlp
```