/// five-point stencil derivative method 
pub fn five_point_derivitive(data: &[f64]) -> Vec<f64>
{
    let mut d = vec![f64::NAN; data.len()];
    if data.len() >= 5 {
        for i in 2..data.len()-2 {
            let mut tmp = data[i-1].mul_add(-8.0, data[i-2]);
            tmp = data[i+1].mul_add(8.0, tmp) - data[i+2];
            d[i] = tmp / 12.0;
        }
    }
    d
}

/// # Calculates the derivative of a Vector
/// * will return a Vector of NaN if `data.len() < 2`
pub fn derivative(data: &[f64]) -> Vec<f64>
{
    let mut d = vec![f64::NAN; data.len()];
    if data.len() >= 3 {
        for i in 1..data.len()-1 {
            d[i] = (data[i+1] - data[i-1]) / 2.0;
        }
    }
    if data.len() >= 2 {
        d[0] = data[1] - data[0];

        d[data.len() - 1] = data[data.len() - 1] - data[data.len() - 2];
    }
    d
}

/// # Calculates the derivative of a Vector
/// * Uses five-point stencil method if more than 4 points are in the vector
/// * Otherwise uses other derivative
/// * will return a Vector of NaN if `data.len() < 2`
pub fn derivative_merged(data: &[f64]) -> Vec<f64>
{
    if data.len() < 5 {
        return derivative(data);
    }
    let mut d = five_point_derivitive(data);
    d[1] = (data[2] - data[0]) / 2.0;
    d[data.len() - 2] = (data[data.len() - 1] - data[data.len() - 3]) / 2.0;

    d[0] = data[1] - data[0];

    d[data.len() - 1] = data[data.len() - 1] - data[data.len() - 2];

    d
}
