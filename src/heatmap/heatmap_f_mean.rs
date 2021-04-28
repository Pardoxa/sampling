use std::{borrow::Borrow, convert::From};

use num_traits::AsPrimitive;
use crate::*;
use average::WeightedMean;


pub struct HeatmapF64Mean<HistX, HistY>
{
    pub(crate) heatmap: HeatmapF64<HistX, HistY>,
    pub(crate) mean: Vec<WeightedMean>
}

impl<HistX, HistY> HeatmapF64Mean<HistX, HistY>
{
    pub fn heatmap(&self) -> &HeatmapF64<HistX, HistY>
    {
        &self.heatmap
    }
}

impl<HistX, HistY> From<HeatmapF64<HistX, HistY>> for HeatmapF64Mean<HistX, HistY>
where HistX: Histogram,
    HistY: Histogram
{
    fn from(heatmap: HeatmapF64<HistX, HistY>) -> Self {
        let x_bins = heatmap.hist_width.bin_count();
        let mean = (0..x_bins)
            .map(|_| WeightedMean::new())
            .collect();

        Self{
            heatmap,
            mean
        }
    }
}

impl<HistX, HistY> HeatmapF64Mean<HistX, HistY>
where HistX: Histogram,
    HistY: Histogram,
{
    pub fn new(hist_x: HistX, hist_y: HistY) -> Self
    {
        let heatmap = HeatmapF64::new(hist_x, hist_y);
        heatmap.into()
    }

    pub fn count_inside_heatmap<X, Y, A, B>(&mut self, x_val: A, y_val: B, weight: f64) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();
        
        let res = self.heatmap.count(x, y, weight);

        if let Ok((x, _)) = res {
            let y_f64 = y.as_();
            if y_f64.is_finite(){
                self.mean[x].add(y_f64, weight);
            }
        }
        res
    }

    pub fn count<X, Y, A, B>(&mut self, x_val: A, y_val: B, weight: f64) -> Result<(usize, usize), HeatmapError>
    where HistX: HistogramVal<X>,
        HistY: HistogramVal<Y>,
        A: Borrow<X>,
        B: Borrow<Y>,
        Y: AsPrimitive<f64>
    {
        let x = x_val.borrow();
        let y = y_val.borrow();

        let res = self.count_inside_heatmap(x, y, weight);
        match res
        {
            Ok(_) => {},
            Err(_) => {
                let y_f64 = y.as_();
                if y_f64.is_finite() {
                    match self.heatmap.hist_width
                        .get_bin_index(x)
                    {
                        Ok(x_bin) => {
                            self.mean[x_bin].add(y_f64, weight);
                        },
                        _ => {}
                    }
                        
                }
            }
        }
        res
    }

    pub fn mean_vec(&self) -> &[WeightedMean]
    {
        &self.mean
    }

    pub fn mean_iter<'a>(&'a self) -> impl Iterator<Item=f64> + 'a
    {
        self.mean
            .iter()
            .map(
                |v|
                {
                    if v.is_empty(){
                        f64::NAN
                    } else 
                    {
                        v.mean()
                    }
                }
            )
    }

    pub fn mean(&self) -> Vec<f64>
    {
        let mut mean = Vec::with_capacity(self.mean.len());

        mean.extend(self.mean_iter());
        mean
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::HistUsizeFast;

    #[test]
    fn average_test()
    {
        let hist_x = HistUsizeFast::new_inclusive(0, 10)
            .unwrap();
        
        let hist_y = hist_x.clone();

        let mut heatmap_mean = HeatmapF64Mean::new(hist_x, hist_y);
    
        for x in 0..=10 {
            for y in 0..=10{
                heatmap_mean.count_inside_heatmap(x, y, 1.0).unwrap();
            }
        }

        for i in heatmap_mean.mean_iter() {
            assert_eq!(i, 5.0);
        }
    }

}