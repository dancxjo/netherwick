#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EyeFrameStats {
    mean_luma: f32,
    non_background_ratio: f32,
}

fn eye_frame_stats(frame: &pete_sensors::EyeFrame) -> EyeFrameStats {
    let pixels = frame.width as usize * frame.height as usize;
    if pixels == 0 {
        return EyeFrameStats::default();
    }
    let mut luma_sum = 0.0f32;
    let mut non_background = 0usize;
    match frame.format {
        EyeFrameFormat::Gray8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
            for pixel in frame.bytes.chunks_exact(3).take(pixels) {
                let (r, g, b) = match frame.format {
                    EyeFrameFormat::Bgr8 => (pixel[2], pixel[1], pixel[0]),
                    _ => (pixel[0], pixel[1], pixel[2]),
                };
                let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
                luma_sum += luma;
                if luma > 0.08 || r.abs_diff(g) > 8 || g.abs_diff(b) > 8 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Yuyv422 | EyeFrameFormat::Uyvy422 => {
            for pair in frame.bytes.chunks_exact(4).take(pixels.div_ceil(2)) {
                let values = match frame.format {
                    EyeFrameFormat::Uyvy422 => [pair[1], pair[3]],
                    _ => [pair[0], pair[2]],
                };
                for value in values {
                    let luma = value as f32 / 255.0;
                    luma_sum += luma;
                    if luma > 0.08 {
                        non_background += 1;
                    }
                }
            }
        }
        EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {}
    }
    EyeFrameStats {
        mean_luma: luma_sum / pixels as f32,
        non_background_ratio: non_background as f32 / pixels as f32,
    }
}

fn encode_eye_data_url(frame: &pete_sensors::EyeFrame) -> (Option<String>, Option<String>) {
    match frame.format {
        EyeFrameFormat::Mjpeg => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&frame.bytes);
            (Some(format!("data:image/jpeg;base64,{encoded}")), None)
        }
        EyeFrameFormat::Rgb8
        | EyeFrameFormat::Bgr8
        | EyeFrameFormat::Gray8
        | EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            let rgb = match eye_frame_to_rgb(frame) {
                Ok(rgb) => rgb,
                Err(error) => return (None, Some(error)),
            };
            let mut png = Vec::new();
            let result = PngEncoder::new(&mut png).write_image(
                &rgb,
                frame.width,
                frame.height,
                ColorType::Rgb8.into(),
            );
            match result {
                Ok(()) => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
                    (Some(format!("data:image/png;base64,{encoded}")), None)
                }
                Err(error) => (None, Some(format!("failed to encode eye PNG: {error}"))),
            }
        }
        EyeFrameFormat::Unknown(ref format) => {
            (None, Some(format!("unsupported eye frame format {format}")))
        }
    }
}

fn bayer8_to_rgb(bytes: &[u8], width: usize, height: usize, format: &EyeFrameFormat) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = bayer_pixel_to_rgb(bytes, width, height, x, y, format);
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

fn bayer_pixel_to_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
) -> (u8, u8, u8) {
    let value = bayer_sample(bytes, width, x, y);
    match bayer_color_at(x, y, format) {
        BayerColor::Red => (
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Green => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Blue => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            value,
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BayerColor {
    Red,
    Green,
    Blue,
}

fn bayer_color_at(x: usize, y: usize, format: &EyeFrameFormat) -> BayerColor {
    let even_x = x % 2 == 0;
    let even_y = y % 2 == 0;
    match format {
        EyeFrameFormat::BayerGrbg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Red,
            (false, true) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerRggb8 => match (even_y, even_x) {
            (true, true) => BayerColor::Red,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerBggr8 => match (even_y, even_x) {
            (true, true) => BayerColor::Blue,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Red,
        },
        EyeFrameFormat::BayerGbrg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Blue,
            (false, true) => BayerColor::Red,
        },
        _ => BayerColor::Green,
    }
}

fn average_bayer_neighbors(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
    colors: &[BayerColor],
) -> u8 {
    let mut sum = 0usize;
    let mut count = 0usize;
    let min_y = y.saturating_sub(1);
    let max_y = (y + 1).min(height.saturating_sub(1));
    let min_x = x.saturating_sub(1);
    let max_x = (x + 1).min(width.saturating_sub(1));
    for ny in min_y..=max_y {
        for nx in min_x..=max_x {
            if nx == x && ny == y {
                continue;
            }
            if colors.contains(&bayer_color_at(nx, ny, format)) {
                sum += bayer_sample(bytes, width, nx, ny) as usize;
                count += 1;
            }
        }
    }
    if count == 0 {
        bayer_sample(bytes, width, x, y)
    } else {
        (sum / count) as u8
    }
}

fn bayer_sample(bytes: &[u8], width: usize, x: usize, y: usize) -> u8 {
    bytes[y * width + x]
}

fn yuyv422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let y0 = pair[0];
        let u = pair[1];
        let y1 = pair[2];
        let v = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn uyvy422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let u = pair[0];
        let y0 = pair[1];
        let v = pair[2];
        let y1 = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn push_yuv_rgb(rgb: &mut Vec<u8>, y: u8, u: u8, v: u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    rgb.push(((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8);
}
