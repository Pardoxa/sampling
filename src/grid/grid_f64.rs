use super::*;

#[derive(Debug, Clone, Copy)]
pub struct Point2D{
    pub x: f64,
    pub y: f64
}

#[derive(Clone)]
pub struct GridF64{
    x_range: GridRangeF64,
    y_range: GridRangeF64,
}

impl GridF64{
    pub fn new(
        x_range: GridRangeF64,
        y_range: GridRangeF64
    ) -> Self
    {
        Self{
            x_range,
            y_range
        }
    }

    pub fn x_range(&self) -> &GridRangeF64
    {
        &self.x_range
    }

    pub fn x_range_iter(&self) -> GridRangeIterF64
    {
        self.x_range.iter()
    }

    pub fn y_range(&self) -> &GridRangeF64
    {
        &self.y_range
    }

    pub fn y_range_iter(&self) -> GridRangeIterF64
    {
        self.y_range.iter()
    }

    /// Checks if a point is contained within the range 
    /// specified by `self`
    /// ## Note
    /// This does not check wether or not the point will 
    /// actually be returned when iterating over the Grid
    pub fn contains(&self, point: &Point2D) -> bool
    {
        self.contains_xy(point.x, point.y)
    }

    pub fn contains_xy(&self, x: f64, y: f64) -> bool
    {
        self.x_range.contains(&x) && self.y_range.contains(&y)
    }

    pub fn grid_point2d_iter(&'_ self) -> impl Iterator<Item=Point2D> + '_
    {
        self.grid_iter()
            .map(|(x, y)| Point2D{x, y})
    }

    pub fn grid_iter(&'_ self) -> impl Iterator<Item=(f64, f64)> + '_
    {
        self.x_range.iter()
            .flat_map(
                |x| 
                {
                    std::iter::repeat(x)
                        .zip(self.y_range.iter())
                }
            )
    }
}

#[cfg(test)]
mod testing
{
    use super::*;

    #[test]
    fn iter_test()
    {
        let x_range = GridRangeF64::new(0.0, 1.0, 3);
        let y_range = GridRangeF64::new(4.0, 3.0, 3);

        let grid = GridF64::new(x_range, y_range);

        let mut iter = grid.grid_iter();

        assert_eq!(Some((0.0, 4.0)), iter.next());
        assert_eq!(Some((0.0, 3.5)), iter.next());
        assert_eq!(Some((0.0, 3.0)), iter.next());
        assert_eq!(Some((0.5, 4.0)), iter.next());
        assert_eq!(Some((0.5, 3.5)), iter.next());
        assert_eq!(Some((0.5, 3.0)), iter.next());
        assert_eq!(Some((1.0, 4.0)), iter.next());
        assert_eq!(Some((1.0, 3.5)), iter.next());
        assert_eq!(Some((1.0, 3.0)), iter.next());
        assert_eq!(None, iter.next());
    }
}