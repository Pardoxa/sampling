use std::{
        fmt,
        io::Write,
        convert::From,
        borrow::*,
        path::Path
    };

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// For labeling the gnuplot plots axis
pub enum Labels{
    /// construct the labels
    FromValues{
        /// minimum value for axis labels
        min: f64,
        /// maximum value for axis labels
        max: f64,
        ///number of tics, should be at least 2
        tics: usize,
    },
    /// use labels 
    FromStrings{
        /// this are the labels
        labels: Vec<String>
    }
}
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// For labeling the gnuplot plots axis
pub struct GnuplotAxis{
    labels: Labels,
    rotation: f32
}

impl GnuplotAxis{
    /// Set the rotation value. 
    /// Tics will be displayed rotated to the right by the requested amount
    pub fn set_rotation(&mut self, rotation_degrees: f32)
    {
        self.rotation = rotation_degrees;
    }

    pub(crate) fn write_tics<W: Write>(&self, mut w: W, num_bins: usize, axis: &str) -> std::io::Result<()>
    {
        match &self.labels {
            Labels::FromValues{min, max, tics} => {
                if min.is_nan() || max.is_nan() || *tics < 2 || num_bins < 2 {
                    Ok(())
                } else {
                    let t_m1 = tics - 1;
                    let difference = (max - min) / t_m1 as f64;
                    
                    let bin_dif = (num_bins - 1) as f64 / t_m1 as f64;
                    write!(w, "set {}tics ( ", axis)?;
                    for i in  0..t_m1 {
                        
                        let val = min + i as f64 * difference;
                        let pos = i as f64 * bin_dif;
                        write!(w, "\"{:#}\" {:e}, ", val, pos)?; 
                    }
                    writeln!(w, "\"{:#}\" {:e} ) rotate by {} right", max,  num_bins - 1, self.rotation)
                }
            }, 
            Labels::FromStrings{labels} => {
                let tics = labels.len();
                match tics {
                    0 => Ok(()),
                    1 => {
                        writeln!(w, "set {}tics ( \"{}\" 0 )", axis, labels[0])
                    },
                    _ => {
                        write!(w, "set {}tics ( ", axis)?;
                        let t_m1 = tics - 1;
                        let bin_dif = (num_bins - 1) as f64 / t_m1 as f64;
                        for (i, lab) in labels.iter().enumerate(){
                            let pos = i as f64 * bin_dif;
                            write!(w, "\"{}\" {:e}, ", lab, pos)?; 
                        }
                        writeln!(w, " ) rotate by {} right", self.rotation)
                    }
                }
            }
        }
        
    }

    /// Create new GnuplotAxis::FromValues
    pub fn new(min: f64, max: f64, tics: usize) -> Self {
        let labels = Labels::FromValues{
            min,
            max,
            tics
        };
        Self { labels, rotation: 0.0 }
    }

    /// Create new GnuplotAxis::Labels
    /// - Vector contains labels used for axis
    pub fn from_labels(labels: Vec<String>) -> Self
    {
        let labels = Labels::FromStrings { labels };
        Self{labels, rotation: 0.0}
    }

    /// Similar to `from_labels`
    /// * Slice of slice is converted to Vector of Strings and `Self::from_labels(vec)` is called
    pub fn from_slice(labels: &[&str]) -> Self {
        let vec = labels.iter()
            .map(|&s| s.into())
            .collect();
        
        Self::from_labels(vec)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Settings for gnuplot
/// * implements default
/// * implements builder pattern for itself
/// # **Safety**
/// 
/// **These gnuplot options are not meant for production!**
/// If you allow arbitrary user input for this, the resulting gnuplot scripts can contain 
/// **arbitrary system calls!** 
/// 
/// Thus calling the resulting gnuplot scripts is not safe, if you have not sanitized the inputs!
/// 
/// This is not an issue if you only create scripts for yourself, i.e., if you are your own user.
pub struct GnuplotSettings{
    /// x label for gnuplot
    pub x_label: String,
    /// how to format the labels of the x axis?
    pub x_axis: Option<GnuplotAxis>,
    /// y label for gnuplot
    pub y_label: String,
    /// how to format the labels of the y axis?
    pub y_axis: Option<GnuplotAxis>,
    /// title for gnuplot
    pub title: String,
    /// which terminal to use for gnuplot
    pub terminal: GnuplotTerminal,

    /// Color palette for heatmap
    pub palette: GnuplotPalette,

    /// Define the cb range if this option is set
    pub cb_range: Option<(f64, f64)>,

    /// # Size of the terminal
    /// * Anything gnuplot accepts (e.g. "2cm, 2.9cm") is acceptable
    /// # Note
    /// the code does not check, if your input for `size` makes any sense
    pub size: String,
}

impl GnuplotSettings {
    /// # Builder pattern - set size of terminal
    /// * Anything gnuplot accepts (e.g. "2cm, 2.9cm") is acceptable
    /// # Note
    /// the code does not check, if your input for `size` makes any sense
    pub fn size<S: Into<String>>(&'_ mut self, size: S) -> &'_ mut Self
    {
        self.size = size.into();
        self
    }

    /// # Builder pattern - set cb_range
    pub fn cb_range(&'_ mut self, range_start: f64, range_end: f64) -> &'_ mut Self
    {
        self.cb_range = Some((range_start, range_end));
        self
    }

    /// # Builder pattern - remove cb_range
    pub fn remove_cb_range(&'_ mut self) -> &'_ mut Self
    {
        self.cb_range = None;
        self
    }

    /// # Builder pattern - set x_label
    pub fn x_label<S: Into<String>>(&'_ mut self, x_label: S) -> &'_ mut Self
    {
        self.x_label = x_label.into();
        self
    }

    pub(crate) fn write_label<W: Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        if !self.x_label.is_empty(){
            writeln!(writer, "set xlabel \"{}\"", self.x_label)?;
        }
        if !self.y_label.is_empty(){
            writeln!(writer, "set ylabel \"{}\"", self.y_label)
        } else {
            Ok(())
        }
    }

    /// # Builder pattern - set y_label
    pub fn y_label<S: Into<String>>(&'_ mut self, y_label: S) -> &'_ mut Self
    {
        self.y_label = y_label.into();
        self
    }

    /// # Builder pattern - set title
    pub fn title<S: Into<String>>(&'_ mut self, title: S) -> &'_ mut Self
    {
        self.title = title.into();
        self
    }

    /// # currently set title
    pub fn get_title(&self) -> &str
    {
        &self.title
    }

    /// # Builder pattern - set terminal
    pub fn terminal(&'_ mut self, terminal: GnuplotTerminal) -> &'_ mut Self
    {
        self.terminal = terminal;
        self
    }

    pub(crate) fn write_terminal<W: Write>(
        &self,
        writer: W
    ) -> std::io::Result<()> {
        self.terminal.write_terminal(writer, &self.size)
    }

    /// # Builder pattern - set color palette
    pub fn palette(&'_ mut self, palette: GnuplotPalette) -> &'_ mut Self
    {
        self.palette = palette;
        self
    }

    /// Create new, default, GnuplotSettings
    pub fn new() -> Self
    {
        Self::default()
    }

    /// Set x_axis - See GnuplotAxis or try it out
    pub fn x_axis(&'_ mut self, axis: GnuplotAxis) -> &'_ mut Self
    {
        self.x_axis = Some(axis);
        self
    }

    /// Remove x_axis
    pub fn remove_x_axis(&'_ mut self) -> &'_ mut Self
    {
        self.x_axis = None;
        self
    }

    /// Set y_axis - See GnuplotAxis or try it out
    pub fn y_axis(&'_ mut self, axis: GnuplotAxis) -> &'_ mut Self
    {
        self.y_axis = Some(axis);
        self
    }

    /// Remove y_axis
    pub fn remove_y_axis(&'_ mut self) -> &'_ mut Self
    {
        self.y_axis = None;
        self
    }

    pub(crate) fn write_axis<W: Write>(&self, mut w: W, num_bins_x: usize, num_bins_y: usize) -> std::io::Result<()>
    {
        if let Some(ax) = self.x_axis.as_ref() {
            ax.write_tics(&mut w, num_bins_x, "x")?;
        }
        if let Some(ax) = self.y_axis.as_ref() {
            ax.write_tics(w, num_bins_y, "y")?;
        }
        Ok(())
    }

    pub(crate) fn write_heatmap_helper1<W>(
        &self, 
        mut writer: W,
        x_len: usize,
        y_len: usize
    ) -> std::io::Result<()>
    where W: Write
    {
        self.write_terminal(&mut writer)?;
       
        self.write_label(&mut writer)?;

        writeln!(writer, "set xrange[-0.5:{}]", x_len as f64 - 0.5)?;
        writeln!(writer, "set yrange[-0.5:{}]", y_len as f64 - 0.5)?;
        if let Some((range_start, range_end)) = self.cb_range{
            writeln!(writer, "set cbrange [{range_start:e}:{range_end:e}]")?;
        }
        if !self.title.is_empty(){
            writeln!(writer, "set title '{}'", self.title)?;
        }

        self.write_axis(
            &mut writer,
            x_len,
            y_len
        )?;

        self.palette.write_palette(&mut writer)?;
        writeln!(writer, "set view map")?;

        writeln!(writer, "set rmargin screen 0.8125\nset lmargin screen 0.175")
    }

    /// # Write a heatmap with the given gnuplot Settings
    /// * `closure` has to write the heatmap. It must write `y_len` rows with `x_len` values each, where the latter values are separated by a space.
    ///   This data will be used for the heatmap.
    /// * `x_len`: The number of entries in each column, that you promise the `closure` will write
    /// * `y_len`: The number of columns you promise that the `closure` will write
    /// # **Safety**
    /// 
    /// **These gnuplot options are not meant for production!**
    /// If you allow arbitrary user input for the gnuplot settings, the resulting gnuplot scripts can contain 
    /// **arbitrary system calls!** 
    /// 
    /// Thus calling the resulting gnuplot scripts is not safe, if you have not sanitized the inputs!
    /// 
    /// This is not an issue if you only create scripts for yourself, i.e., if you are your own user.
    pub fn write_heatmap<F, W>(
        &self, 
        mut writer: W, 
        closure: F,
        x_len: usize,
        y_len: usize
    ) -> std::io::Result<()>
    where W: Write,
        F: FnOnce (&mut W) -> std::io::Result<()>
    {
        self.write_heatmap_helper1(
            &mut writer,
            x_len,
            y_len
        )?;
        writeln!(writer, "$data << EOD")?;
        closure(&mut writer)?;

        writeln!(writer, "EOD")?;

        writeln!(writer, "splot $data matrix with image t \"{}\" ", &self.title)?;

        self.terminal.finish(&mut writer)
    }

    /// Same as write_heatmap but it assumes that the heatmap 
    /// matrix is available in the file "heatmap"
    /// # **Safety**
    /// 
    /// **These gnuplot options are not meant for production!**
    /// If you allow arbitrary user input for the gnuplot settings, the resulting gnuplot scripts can contain 
    /// **arbitrary system calls!** 
    /// 
    /// Thus calling the resulting gnuplot scripts is not safe, if you have not sanitized the inputs!
    /// 
    /// This is not an issue if you only create scripts for yourself, i.e., if you are your own user.
    pub fn write_heatmap_external_matrix<W, P>(
        &self,
        mut writer: W,
        matrix_width: usize,
        matrix_height: usize,
        matrix_path: P
    ) -> std::io::Result<()> 
    where W: Write,
        P: AsRef<Path>
    {
        self.write_heatmap_helper1(
            &mut writer,
            matrix_width,
            matrix_height
        )?;

        writeln!(
            writer, 
            "splot \"{}\" matrix with image t \"{}\" ", 
            matrix_path.as_ref().to_string_lossy(),
            &self.title
        )?;

        self.terminal.finish(&mut writer)
    }
}

impl Default for GnuplotSettings{
    fn default() -> Self {
        Self{
            x_label: "".to_owned(),
            y_label: "".to_owned(),
            title: "".to_owned(),
            terminal: GnuplotTerminal::Empty,
            palette: GnuplotPalette::PresetHSV,
            x_axis: None,
            y_axis: None,
            size: "7.4cm, 5cm".into(),
            cb_range: None
        }
    }
}

#[derive(Debug,Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// Implements color palett from <https://arxiv.org/abs/1108.5083>
///
/// What's so good about this palett? It is monotonically increasing in perceived brightness.
/// That means, it is well suited for being printed in black and white. 
///
/// ```
/// use sampling::heatmap::*;
/// let mut params = CubeHelixParameter::default();
/// params.rotation(1.3)
///         .gamma(1.1)
///         .start_color(0.3)
///         .reverse(true);
/// ```
pub struct CubeHelixParameter{
    hue: f32,       // hue intensity, valid values 0 <= hue <= 1
    r:  f32,        // rotation in color space. Typical values: -1.5 <= r <= 1.5
    s: f32,         // starting color, valid values 0 <= s <= 1
    gamma: f32,     // gamma < 1 emphasises low intensity values, gamma > 1 high intensity ones
    low: f32,       // lowest value for grayscale. 0 <= low < 1 and low < high
    high: f32,      // highest value for grayscale. 0< high <= 1 and low < high
    reverse: bool   // reverse cbrange?
}

fn valid(v: f32) -> bool
{
    (0.0..=1.0).contains(&v)
}

impl CubeHelixParameter {
    /// # Builder pattern - set start color
    /// Will panic if the following is false: 0.0 <= s <= 1.0
    pub fn start_color(&mut self, s: f32) -> &mut Self 
    {
        if valid(s) {
            self.s = s;
            return self;
        }
        panic!("Invalid value for starting color! The following has to be true: 0.0 <= s <= 1.0 - you used: s={}", s)
    }

    /// # Builder pattern - set gamma
    /// 
    /// |gamma| < 1 emphasises low intensity values, |gamma| > 1 high intensity ones
    ///
    /// gamma has to be finite - will panic otherwise
    pub fn gamma(&mut self, gamma: f32) -> &mut Self
    {
        if gamma.is_finite(){
            self.gamma = gamma.abs();
            return self;
        }
        panic!("Gamma has to be finite. You used: {}", gamma)
    }


    /// # Builder pattern - set reverse
    /// reverse: Reverse the cbrange?
    pub fn reverse(&mut self, reverse: bool) -> &mut Self
    {
        self.reverse = reverse;
        self
    }

    /// # Builder pattern - set low and high value
    /// default: low = 0.0, high = 1.0
    ///
    /// Maps grayscale range from [0.0, 1.0] -> [low, high].
    /// These are the brightness values used for calculating the palette later on.
    /// 
    /// # Safety
    /// will panic if
    /// * `low` >= `high`
    /// * `low` < 0
    /// * `low` >= 1
    /// * `high` <= 0
    /// * `high` > 1
    pub fn low_high(&mut self, low: f32, high: f32) -> &mut Self
    {
        if low < high && valid(low) && valid(high) {
            self.low = low;
            self.high = high;
            return self;
        }
        panic!("Invalid values of low and high. The following has to be true: 0.0 <= low < high <= 1.0. You used: low {} high {}", low, high)
    }


    /// #Set hue intensity. Builder pattern
    /// Valid values are 0.0 <= hue <= 1.0.
    /// **Important** Will panic on invalid hue values!
    pub fn hue(&mut self, hue: f32) -> &mut Self
    {
        if valid(hue) {
            self.hue = hue;
        } else {
            panic!("Invalid hue value! Hue value has to be 0.0 <= hue <= 1.0, you used {}", hue)
        }
        self
    }

    /// #Set rotation. Builder pattern
    /// Rotation in color space. The higher the value, the quicker the colors will change in the palett.
    ///
    /// Normally the range used is -1.5 <= rotation <= 1.5. Invalid values are Nan, or ±Infinity
    /// **Important** Will panic on invalid rotation values!
    pub fn rotation(&mut self, rotation: f32) -> &mut Self
    {
        if rotation.is_finite(){
            self.r = rotation;
        }else {
            panic!("Invalid rotation value! Rotation value has to be finite, you used {}", rotation)
        }
        self
    }

    /// Calculate color from gray value.
    /// Gray value should be in the interval [0.0,1.0].
    /// 
    /// Will return `[red, green, blue]`, where red, green and blue are in [0.0, 1.0],
    /// will return \[0,0,0\] for NAN gray value.
    pub fn rgb_from_gray(&self, gray: f32) -> [f32; 3]
    {
        if gray.is_nan() {
            return [0.0, 0.0, 0.0];
        }
        let mut lambda = gray.clamp(0.0, 1.0);
        if self.reverse {
            lambda = 1.0 - lambda;
        }
        lambda = self.low + (self.high - self.low) * lambda;    // map [0,1] -> [low, high]
        let lg = lambda.powf(self.gamma);
        let phi = 2.0 * (self.s / 3.0 + self.r * lambda) * std::f32::consts::PI;
        let a = self.hue * lg * (1.0 - lg) * 0.5;

        let (s_phi, c_phi) = phi.sin_cos();

        [
            a * (-0.14861 * c_phi + 1.78277 * s_phi)+ lg,   // red
            lg + a * (-0.29227 * c_phi - 0.90649 * s_phi),  // green
            lg +  a * c_phi * 1.97294                       // blue
        ]
    }

    /// * Calculate color from gray value.
    /// * Gray value should be in the interval [0.0,1.0].
    /// * will return `ColorRgb::new(0,0,0)` for NAN gray value
    /// 
    /// will return corresponding (approximate) [`ColorRgb`](crate::heatmap::ColorRGB)
    pub fn approximate_color_rgb(&self, gray: f32) -> ColorRGB
    {
        let color = self.rgb_from_gray(gray);
        let color = color
            .map(
                |val| 
                (val * 255.0)
                    .clamp(0.0,255.0)
                    .floor() as u8
            );
        
        ColorRGB::new_from_array(&color)
    }

    /// Converts `self` into the corresponding enum of [`GnuplotPallet`](crate::heatmap::GnuplotPalette)
    pub fn into_gnuplot_palette(self) -> GnuplotPalette
    {
        self.into()
    }
}

impl Default for CubeHelixParameter {
    fn default() -> Self 
    {
        Self{
            hue: 1.0,
            r: 1.2,
            low: 0.0,
            high: 1.0,
            reverse: false,
            gamma: 1.0,
            s: 0.1
        }    
    }
}


/// # RGB value
/// * stores a color in RGB space
/// * default color is black `[0,0,0]`
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct ColorRGB{
    /// The red part
    pub red: u8,
    /// The green part
    pub green: u8,
    /// The blue part
    pub blue: u8
}

impl Default for ColorRGB{
    fn default() -> Self 
    {
        Self::new(0,0,0)
    }
}

impl ColorRGB{
    /// # Create a new color
    pub fn new(red: u8, green: u8, blue: u8) -> Self
    {
        Self{
            red,
            green, 
            blue
        }
    }

    /// # Create color from an array
    /// * `color[0]` -> red
    /// * `color[1]` -> green
    /// * `color[2]` -> blue
    pub fn new_from_array(color: &[u8;3]) -> Self
    {
        Self{
            red: color[0],
            green: color[1],
            blue: color[2]
        }
    }

    /// # convert color to array,
    /// will return `[red, green, blue]`
    pub fn to_array(&self) -> [u8;3]
    {
        [self.red, self.green, self.blue]
    }

    /// # Turn into hex representation
    /// ```
    /// use sampling::heatmap::ColorRGB;
    /// 
    /// let color = ColorRGB::new(0,0,0);
    /// let hex = color.to_hex();
    /// assert_eq!(&hex, "#000000");
    /// 
    /// let color = ColorRGB::new_from_array(&[255,255,255]);
    /// let hex = color.to_hex();
    /// assert_eq!(&hex, "#FFFFFF");
    /// ```
    pub fn to_hex(&self) -> String
    {
        let mut s = String::new();
        self.fmt_hex(&mut s)
            .unwrap();
        s
    }

    /// # Write hex representation to a fmt writer
    /// * similar to [`to_hex`](crate::heatmap::ColorRGB::to_hex), but writes to fmt writer instead
    pub fn fmt_hex<W: fmt::Write>(&self, mut writer: W) -> Result<(), fmt::Error>
    {
        write!(
            writer,
            "#{:02X?}{:02X?}{:02X?}",
            self.red,
            self.green,
            self.blue
        )
    }

    /// # Write hex representation to a io writer
    /// * similar to [`to_hex`](crate::heatmap::ColorRGB::to_hex), but writes to io writer instead
    pub fn write_hex<W: Write>(&self, mut writer: W) -> Result<(), std::io::Error>
    {
        write!(
            writer,
            "#{:02X?}{:02X?}{:02X?}",
            self.red,
            self.green,
            self.blue
        )
    }
}

/// # A color palette in RGB space
/// * used for [GnuplotPalette](crate::heatmap::GnuplotPalette)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct PaletteRGB{
    colors: Vec<ColorRGB>
}

impl PaletteRGB{
    /// # Initialize Palette
    /// Note: Palette needs at least two colors,
    /// therefore `None` will be returned if the 
    /// `colors` has a length less than 2
    pub fn new(colors: Vec<ColorRGB>) -> Option<Self>
    {
        if colors.len() < 2 
        {
            return None;
        }
        Some(
            Self{
                colors
            }
        )
    }

    /// # add a color to the palette
    pub fn add_color(&mut self, color: ColorRGB){
        self.colors.push(color)
    }

    /// # write string to define this palette in gnuplot to fmt writer
    pub fn fmt_palette<W: fmt::Write>(&self, mut writer: W) -> Result<(), fmt::Error>
    {
        write!(writer, "set palette defined ( 0 \"")?;
        self.colors[0].fmt_hex(&mut writer)?;
        write!(writer, "\"")?;

        for (color, index) in self.colors.iter().skip(1).zip(1..)
        {
            write!(writer, ", {} \"", index)?;
            color.fmt_hex(&mut writer)?;
            write!(writer, "\"")?;

        }
        write!(writer, " )")  
    }

    /// # write string to define this palette in gnuplot to io writer
    pub fn write_palette<W: std::io::Write>(&self, mut writer: W) -> Result<(), std::io::Error>
    {
        write!(writer, "set palette defined ( 0 \"")?;
        self.colors[0].write_hex(&mut writer)?;
        write!(writer, "\"")?;

        for (color, index) in self.colors.iter().skip(1).zip(1..)
        {
            write!(writer, ", {} \"", index)?;
            color.write_hex(&mut writer)?;
            write!(writer, "\"")?;

        }
        write!(writer, " )")
    }

    /// Converts `self` into the corresponding enum of [`GnuplotPallet`](crate::heatmap::GnuplotPalette)
    pub fn into_gnuplot_palette(self) -> GnuplotPalette
    {
        self.into()
    }
}

/// # Defines gnuplot point
/// * Note that most of the fields are public and
///   can be accessed directly
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
pub struct GnuplotPointSettings
{
    /// Color of the point
    pub color: ColorRGB,
    size: f32,
    /// should the point have a frame?
    pub frame: bool,
    /// Which color should the frame be?
    pub frame_color: ColorRGB,
    /// Entry for legend
    legend: String
}

impl GnuplotPointSettings{
    /// Create a new instance of GnuplotPointSettings
    /// - same as GnuplotPointSettings::default()
    pub fn new() -> Self
    {
        Self::default()
    }

    /// # Choose the color for the point
    /// * default color is blue
    pub fn color(&mut self, color: ColorRGB) -> &mut Self
    {
        self.color = color;
        self
    }

    /// # Choose the size of the point
    /// * size has to be finite
    /// * size has to be >= 0.0
    /// 
    /// Otherwise the old size will be silently kept
    pub fn size(&mut self, size: f32) -> &mut Self
    {
        if size.is_finite() && size >= 0.0
        {
            self.size = size;
        }
        self
    }

    /// # Get the point size
    /// Note: allmost all other fields are public!
    pub fn get_size(&self) -> f32
    {
        self.size
    }

    /// Should there be a frame around the point ?
    /// This is good for better visibility if you do not know the color
    /// of the background, or the background color changes
    pub fn frame(&mut self, active: bool) -> &mut Self
    {
        self.frame = active;
        self
    }

    /// # Which color should the frame have?
    /// *default color is black
    pub fn frame_color(&mut self, color: ColorRGB) -> &mut Self
    {
        self.frame_color = color;
        self
    }

    /// # Change the legend entry
    /// * This will be the title of the legend for this point(s)
    /// * will be set to "Invalid character encountered" if it contains a " or newline character
    pub fn legend<S: Into<String>>(&mut self, legend: S) -> &mut Self
    {
        let s = legend.into();
        if s.contains('\"') || s.contains('\n')
        {
            self.legend = "Invalid character encountered".to_owned();
            self
        } else {
            self.legend = s;
            self
        }
    }

    /// # Get entry for legend
    /// This will be the title of the legend for this point(s)
    pub fn get_legend(&self) -> &str
    {
        &self.legend
    }

    #[allow(dead_code)]
    pub(crate) fn frame_size(&self) -> f32
    {
        let size = self.size * 1.14;
        if size < 0.01 {
            0.01
        } else {
            size
        }
    }
}

impl Default for GnuplotPointSettings
{
    fn default() -> Self 
    {
        Self{
            color: ColorRGB::new(0,0,255),
            size: 0.5,
            frame: true,
            frame_color: ColorRGB::default(),
            legend: "".into()
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// defines presets for different color palettes
pub enum GnuplotPalette{
    /// Use preset HSV palette
    PresetHSV,
    /// Use preset RGB palette, i.e., the 
    /// default palette of gnuplot
    PresetRGB,
    /// Use a CubeHelix palette
    /// 
    /// What makes this palette special is,
    /// that, if it is converted to black and white,
    /// it will be monotonically increasing in perceived brightness
    /// (or monotonically decreasing, if you reverse the palette),
    /// which is nice for heatmaps
    ///
    /// For more info see [`CubeHelixParameter`](crate::heatmap::CubeHelixParameter)
    CubeHelix(CubeHelixParameter),
    /// Define a palette in RGB space
    RGB(PaletteRGB),
}

impl From<PaletteRGB> for GnuplotPalette{
    fn from(palette: PaletteRGB) -> Self 
    {
        GnuplotPalette::RGB(palette)  
    }
}

impl From<CubeHelixParameter> for GnuplotPalette
{
    fn from(parameter: CubeHelixParameter) -> Self
    {
        GnuplotPalette::CubeHelix(parameter)
    }
}

impl GnuplotPalette{
    pub(crate) fn write_palette<W: Write>(&self, mut writer: W) -> std::io::Result<()>
    {
        match self {
            Self::PresetHSV => {
                writeln!(writer, "set palette model HSV")?;
                writeln!(writer, "set palette negative defined  ( 0 0 1 0, 2.8 0.4 0.6 0.8, 5.5 0.83 0 1 )")
            },
            Self::PresetRGB => Ok(()),
            Self::CubeHelix(helix) => {
                writeln!(writer, "# Parameter for color palett")?;
                writeln!(writer, "hue={}     # hue intensity, valid values 0 <= hue <= 1", helix.hue)?;
                writeln!(writer, "r={}       # rotation in color space. Typical values: -1.5 <= r <= 1.5", helix.r)?;
                writeln!(writer, "s={}       # starting color, valid values 0 <= s <= 1", helix.s)?;
                writeln!(writer, "gamma={}   # gamma < 1 emphasizes low intensity values, gamma > 1 high intensity ones", helix.gamma)?;
                writeln!(writer, "low={}     # lowest value for grayscale. 0 <= low < 1 and low < high", helix.low)?;
                writeln!(writer, "high={}    # highest value for grayscale. 0< high <= 1 and low < high", helix.high)?;
                let s = if helix.reverse {
                    "1"
                } else {
                    "0"
                };
                writeln!(writer, "reverse={}   # set to 1 for reverse cbrange, set to 0 for original cbrange", s)?;
                writeln!(writer, "\n\nlg(lambda)=lambda**gamma
phi(lambda)=2.0 * (s/3.0 + r * lambda) * pi
a(lambda)=hue*lg(lambda)*(1.0-lg(lambda)) * 0.5

red(lambda)=a(lambda)*(-0.14861*cos(phi(lambda)) + 1.78277 * sin(phi(lambda))) + lg(lambda)
green(lambda)=lg(lambda) + a(lambda)*(-0.29227 * cos(phi(lambda)) - 0.90649 * sin(phi(lambda)))
blue(lambda)=lg(lambda) + a(lambda) * cos(phi(lambda)) * 1.97294

rev(x)=reverse?1-x:x    # reverse grayscale

map(x)=low+(high-low)*rev(x)

set palette functions red(map(gray)), green(map(gray)), blue(map(gray))\n")
            },
            Self::RGB(palette) => {
                {
                    palette.write_palette(&mut writer)?;
                    writeln!(writer)
                }
            }
        }
        
    }
}




#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Options for choosing gnuplot Terminal
pub enum GnuplotTerminal{
    /// # Use EpsLatex as terminal in gnuplot
    /// * The String here is the output name, i.e., the filepath of the output of gnuplot (without the .tex)
    ///   Only alphanumeric characters, space and underscore are allowed,
    ///   all other characters will be ignored
    /// * the created gnuplot script assumes, you have `latexmk` installed
    /// * if you do not have latexmk, you can still use this, but you have to manually edit the 
    ///   gnuplot scrip later on
    /// * gnuplot script will create `.tex` file and `.pdf` file created from the tex file
    /// 
    /// ## WARNING
    /// The created gnuplot script will contain a system call to latexmk, such that a pdf file is generated from a tex file
    /// without the need for calling latexmk afterwards. 
    EpsLatex(String),
    /// # Use pdf as gnuplot terminal
    /// * gnuplot script will create a `.pdf` file
    /// * The String here is the output name, i.e., the filepath of the output of gnuplot (without the .pdf)
    PDF(String),
    /// # Does not specify a terminal
    Empty,
}

fn get_valid_filename(name: &str) -> String 
{
    name.chars()
        .filter(
            |c|
            {
                c.is_alphabetic() || *c == ' ' || *c == '_'
            }
        ).take(255)
        .collect()
}

impl GnuplotTerminal{
    pub(crate) fn write_terminal<W: Write>(
        &self,
        mut writer: W,
        size: &str
    ) -> std::io::Result<()>
    {
        writeln!(writer, "reset session")?;
        writeln!(writer, "set encoding utf8")?;

        let size = if size.is_empty(){
            size.to_owned()
        } else {
            format!(" size {}", size)
        };

        match self{
            Self::EpsLatex(name) => {
                let name = get_valid_filename(name);
                writeln!(writer, "set t epslatex 9 standalone color{} header \"\\\\usepackage{{amsmath}}\\n\"\nset font \",9\"", size)?;
                writeln!(writer, "set output \"{name}.tex\"")
            },
            Self::PDF(name) => {
                let name = get_valid_filename(name);
                writeln!(writer, "set t pdf {}", size)?;
                writeln!(writer, "set output \"{name}.pdf\"")
            },
            Self::Empty => Ok(())
        }
    }

    pub(crate) fn finish<W: Write>(&self, mut w: W) -> std::io::Result<()>
    {
        match self {
            Self::EpsLatex(name) => {
                let name = get_valid_filename(name);
                writeln!(w, "set output")?;
                write!(w, "system('latexmk {name}.tex")?;
                writeln!(w, " -pdf -f')")
            },
            Self::PDF(_) => {
                writeln!(w, "set output")
            },
            _ => Ok(())
        }
    } 
}