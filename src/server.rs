use std::sync::Arc;

use candle_core::{Error, Tensor};
use tokio::net::TcpListener;

use crate::io::{read_numpy, write_numpy};

/// Runs a server that accepts numpy arrays and returns the result of a forward pass.
///
/// # Arguments
///
/// * `addr` - The address to bind to.
/// * `model` - The model to run as an Arc.
/// * `net_forward` - The function that runs the forward pass. This should accept
/// a reference to the model and a tensor input and should return a tensor.
pub async fn run_server<M>(
    addr: &str,
    model: Arc<M>,
    net_forward: fn(&M, Tensor) -> Result<Tensor, Error>,
) -> Result<(), Error>
where
    M: Sync + Send + 'static,
{
    let listener = TcpListener::bind(addr).await.expect("Failed to bind.");

    while let Ok((mut socket, _)) = listener.accept().await {
        // get a cloned reference of the weights
        let model_clone = Arc::clone(&model);

        tokio::spawn(async move {
            let (mut reader, mut writer) = socket.split();
            let buf_reader = tokio::io::BufReader::new(&mut reader);

            // read array from the stream
            let input_data = read_numpy(buf_reader)
                .await
                .expect("error reading numpy array");

            // forward pass
            let x = net_forward(&*model_clone, input_data).expect("error making forward pass");

            // write array to the stream
            write_numpy(&x, &mut writer)
                .await
                .expect("error writing numpy array");
        });
    }

    Ok(())
}
