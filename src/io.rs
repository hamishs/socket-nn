/// Module to read and write `numpy` arrays to the stream.
/// Based on `candle_core::npy`.
use candle_core::{DType, Device, Error, Result, Shape, Tensor};
use std::collections::HashMap;
use std::marker::Unpin;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const NPY_MAGIC_STRING: &[u8] = b"\x93NUMPY";

/// Read a `numpy` array from the stream and conver to a `Tensor`.
pub async fn read_numpy<T>(mut reader: T) -> Result<Tensor>
where
    T: AsyncReadExt + Unpin,
{
    let header = read_header(&mut reader).await?;
    let header = Header::parse(&header)?;
    if header.fortran_order {
        return Err(Error::Npy("fortran order not supported".to_string()));
    }
    let shape = header.shape();
    let mut arr: Vec<f64> = vec![];
    let mut data = [0u8; std::mem::size_of::<f64>()];
    for _ in 0..shape.elem_count() {
        reader.read_exact(&mut data).await?;
        let f = f64::from_le_bytes(data);
        arr.push(f);
    }
    Tensor::from_vec(arr, shape, &Device::Cpu)
}

/// Write a `Tensor` to the stream in `numpy` array format.
pub async fn write_numpy<T>(tensor: &Tensor, f: &mut T) -> Result<()>
where
    T: AsyncWriteExt + Unpin,
{
    let header = Header {
        descr: tensor.dtype(),
        fortran_order: false,
        shape: tensor.dims().to_vec(),
    };
    let mut header = header.to_string()?;
    let pad = 16 - (NPY_MAGIC_STRING.len() + 5 + header.len()) % 16;
    for _ in 0..pad % 16 {
        header.push(' ')
    }
    header.push('\n');

    let mut payload = Vec::new();
    payload.extend_from_slice(NPY_MAGIC_STRING);
    payload.extend_from_slice(&[1u8, 0u8]);
    payload.extend_from_slice(&[(header.len() % 256) as u8, (header.len() / 256) as u8]);
    payload.extend_from_slice(header.as_bytes());

    let mut value_bytes = Vec::new();
    let vs = tensor.flatten_all()?;
    for v in vs.to_vec1::<f64>()? {
        value_bytes.extend_from_slice(&v.to_le_bytes());
    }
    payload.extend_from_slice(&value_bytes);

    f.write_all(&payload).await?;

    Ok(())
}

async fn read_header<T>(reader: &mut T) -> Result<String>
where
    T: AsyncReadExt + Unpin,
{
    let mut magic_string = vec![0u8; NPY_MAGIC_STRING.len()];
    reader.read_exact(&mut magic_string).await?;
    if magic_string != NPY_MAGIC_STRING {
        return Err(Error::Npy("magic string mismatch".to_string()));
    }
    let mut version = [0u8; 2];
    reader.read_exact(&mut version).await?;
    let header_len_len = match version[0] {
        1 => 2,
        2 => 4,
        otherwise => return Err(Error::Npy(format!("unsupported version {otherwise}"))),
    };
    let mut header_len = vec![0u8; header_len_len];
    reader.read_exact(&mut header_len).await?;
    let header_len = header_len
        .iter()
        .rev()
        .fold(0_usize, |acc, &v| 256 * acc + v as usize);
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).await?;
    Ok(String::from_utf8_lossy(&header).to_string())
}

#[derive(Debug, PartialEq)]
struct Header {
    descr: DType,
    fortran_order: bool,
    shape: Vec<usize>,
}

impl Header {
    fn shape(&self) -> Shape {
        Shape::from(self.shape.as_slice())
    }

    fn to_string(&self) -> Result<String> {
        let fortran_order = if self.fortran_order { "True" } else { "False" };
        let mut shape = self
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let descr = match self.descr {
            DType::BF16 => Err(Error::Npy("bf16 is not supported".into()))?,
            DType::F16 => "f2",
            DType::F32 => "f4",
            DType::F64 => "f8",
            DType::U32 => "u4",
            DType::U8 => "u1",
        };
        if !shape.is_empty() {
            shape.push(',')
        }
        Ok(format!(
            "{{'descr': '<{descr}', 'fortran_order': {fortran_order}, 'shape': ({shape}), }}"
        ))
    }

    // Hacky parser for the npy header, a typical example would be:
    // {'descr': '<f8', 'fortran_order': False, 'shape': (128,), }
    fn parse(header: &str) -> Result<Header> {
        let header =
            header.trim_matches(|c: char| c == '{' || c == '}' || c == ',' || c.is_whitespace());

        let mut parts: Vec<String> = vec![];
        let mut start_index = 0usize;
        let mut cnt_parenthesis = 0i64;
        for (index, c) in header.chars().enumerate() {
            match c {
                '(' => cnt_parenthesis += 1,
                ')' => cnt_parenthesis -= 1,
                ',' => {
                    if cnt_parenthesis == 0 {
                        parts.push(header[start_index..index].to_owned());
                        start_index = index + 1;
                    }
                }
                _ => {}
            }
        }
        parts.push(header[start_index..].to_owned());
        let mut part_map: HashMap<String, String> = HashMap::new();
        for part in parts.iter() {
            let part = part.trim();
            if !part.is_empty() {
                match part.split(':').collect::<Vec<_>>().as_slice() {
                    [key, value] => {
                        let key = key.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let value = value.trim_matches(|c: char| c == '\'' || c.is_whitespace());
                        let _ = part_map.insert(key.to_owned(), value.to_owned());
                    }
                    _ => return Err(Error::Npy(format!("unable to parse header {header}"))),
                }
            }
        }
        let fortran_order = match part_map.get("fortran_order") {
            None => false,
            Some(fortran_order) => match fortran_order.as_ref() {
                "False" => false,
                "True" => true,
                _ => return Err(Error::Npy(format!("unknown fortran_order {fortran_order}"))),
            },
        };
        let descr = match part_map.get("descr") {
            None => return Err(Error::Npy("no descr in header".to_string())),
            Some(descr) => {
                if descr.is_empty() {
                    return Err(Error::Npy("empty descr".to_string()));
                }
                if descr.starts_with('>') {
                    return Err(Error::Npy(format!("little-endian descr {descr}")));
                }
                // the only supported types in tensor are:
                //     float64, float32, float16,
                //     complex64, complex128,
                //     int64, int32, int16, int8,
                //     uint8, and bool.
                match descr.trim_matches(|c: char| c == '=' || c == '<' || c == '|') {
                    "e" | "f2" => DType::F16,
                    "f" | "f4" => DType::F32,
                    "d" | "f8" => DType::F64,
                    // "i" | "i4" => DType::S32,
                    // "h" | "i2" => DType::S16,
                    // "b" | "i1" => DType::S8,
                    "B" | "u1" => DType::U8,
                    "I" | "u4" => DType::U32,
                    "?" | "b1" => DType::U8,
                    // "F" | "F4" => DType::C64,
                    // "D" | "F8" => DType::C128,
                    descr => return Err(Error::Npy(format!("unrecognized descr {descr}"))),
                }
            }
        };
        let shape = match part_map.get("shape") {
            None => return Err(Error::Npy("no shape in header".to_string())),
            Some(shape) => {
                let shape = shape.trim_matches(|c: char| c == '(' || c == ')' || c == ',');
                if shape.is_empty() {
                    vec![]
                } else {
                    shape
                        .split(',')
                        .map(|v| v.trim().parse::<usize>())
                        .collect::<std::result::Result<Vec<_>, _>>()?
                }
            }
        };
        Ok(Header {
            descr,
            fortran_order,
            shape,
        })
    }
}
