use std::io::Write;
use super::*;

#[derive(Debug, Clone)]
pub struct  Point3D
{
    pub x: f64,
    pub y: f64,
    pub z: f64
}

#[derive(Clone)]
pub struct GridMapF64
{
    grid: GridF64,
    values: Vec<f64>
}

impl GridMapF64{

    pub fn from_fn<F>(grid: GridF64, mapper: F) -> Self
    where F: FnMut (Point2D) -> f64
    {
        let vec: Vec<_> = grid.grid_point2d_iter()
            .map(mapper)
            .collect();
        Self{
            grid,
            values: vec
        }
    }

    pub fn iter(&'_ self) -> impl Iterator<Item=((f64, f64), f64)> + '_
    {
        self.grid.grid_iter()
            .zip(self.values.iter().copied())
    }

    pub fn iter_point3d(&'_ self) -> impl Iterator<Item=Point3D> + '_
    {
        self.iter()
            .map(
                |((x, y), z)|
                Point3D{x, y, z}
            )
    }

    pub fn write<W>(&self, mut writer: W) -> std::io::Result<()>
    where W: Write
    {
        writeln!(writer, "#X Y Z")?;

        let iter = self.grid.x_range()
            .iter()
            .flat_map(
                |x| 
                {
                    std::iter::once(true)
                        .chain(std::iter::repeat(false))
                        .zip(std::iter::repeat(x))
                        .zip(self.grid.y_range().iter())
                }
            ).zip(self.values.iter());
        for (((new_line, x), y), z) in iter
        {
            if new_line {
                writeln!(writer)?;
            }
            writeln!(writer, "{:E} {:E} {:E}", x, y, z)?;
        }
        Ok(())
    }

    /// # Create Something that can be plotted with Gnuplot!
    /// In the gnuplotterminal (opend by writing gnuplot in the Terminal) write `load filename` to load
    pub fn write_gnuplot<W>(&self, mut writer: W) -> std::io::Result<()>
    where W: Write
    {
        writeln!(writer, "$data << EOD")?;
        self.write(&mut writer)?;
        writeln!(writer, "EOD")?;
        writeln!(writer, "splot $data with lines")
    }
}

#[cfg(test)]
mod grid_val_tests
{
    use super::*;
    use std::fs::File;
    use std::io::BufWriter;

    #[test]
    fn writer()
    {
        let file = File::create("grid_test.gp")
            .unwrap();
        let buf = BufWriter::new(file);

        let x_range = GridRangeF64::new(1.0,-1.0, 9);
        let y_range = GridRangeF64::new(-2.0, 2.0, 15);

        let grid = GridF64::new(x_range, y_range);
        let grid_map = GridMapF64::from_fn(grid, |point| point.x * point.x + point.y * point.y);
        grid_map.write_gnuplot(buf).unwrap();
    }
}