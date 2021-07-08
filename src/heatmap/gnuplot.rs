use std::borrow::*;
use std::io::Write;
use std::default::Default;

#[cfg(feature = "serde_support")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// For labeling the gnuplot plots axis
pub enum GnuplotAxis{
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
    Labels{
        /// this are the labels
        labels: Vec<String>
    }
}

impl GnuplotAxis{
    pub(crate) fn write_tics<W: Write>(&self, mut w: W, num_bins: usize, axis: &str) -> std::io::Result<()>
    {
        match self {
            Self::FromValues{min, max, tics} => {
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
                    writeln!(w, "\"{:#}\" {:e} )", max,  num_bins - 1)
                }
            }, 
            Self::Labels{labels} => {
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
                        writeln!(w, " )")
                    }
                }
            }
        }
        
    }

    /// Create new GnuplotAxis::FromValues
    pub fn new(min: f64, max: f64, tics: usize) -> Self {
        Self::FromValues{
            min,
            max,
            tics
        }
    }

    /// Create new GnuplotAxis::Labels
    /// - Vector contains labels used for axis
    pub fn from_labels(labels: Vec<String>) -> Self
    {
        Self::Labels{
            labels
        }
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
pub struct GnuplotSettings{
    /// x label for gnuplog
    pub x_label: String,
    /// how to format the lables of the x axis?
    pub x_axis: Option<GnuplotAxis>,
    /// y label for gnuplot
    pub y_label: String,
    /// how to format the lables of the y axis?
    pub y_axis: Option<GnuplotAxis>,
    /// title for gnuplot
    pub title: String,
    /// which terminal to use for gnuplot
    pub terminal: GnuplotTerminal,

    /// Color pallet for heatmap
    pub pallet: GnuplotPallet,
}

impl GnuplotSettings {
    /// # Builder pattern - set x_label
    pub fn x_label<'a, S: Into<String>>(&'a mut self, x_label: S) -> &'a mut Self
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
    pub fn y_label<'a, S: Into<String>>(&'a mut self, y_label: S) -> &'a mut Self
    {
        self.y_label = y_label.into();
        self
    }

    /// # Builder pattern - set title
    pub fn title<'a, S: Into<String>>(&'a mut self, title: S) -> &'a mut Self
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
    pub fn terminal<'a>(&'a mut self, terminal: GnuplotTerminal) -> &'a mut Self
    {
        self.terminal = terminal;
        self
    }

    pub(crate) fn terminal_str(&self) -> &'static str {
        self.terminal.terminal_str()
    }

    /// # Builder pattern - set color pallet
    pub fn pallet<'a>(&'a mut self, pallet: GnuplotPallet) -> &'a mut Self
    {
        self.pallet = pallet;
        self
    }

    /// Create new, default, GnuplotSettings
    pub fn new() -> Self
    {
        Self::default()
    }

    /// Set x_axis - See GnuplotAxis or try it out
    pub fn x_axis<'a>(&'a mut self, axis: GnuplotAxis) -> &'a mut Self
    {
        self.x_axis = Some(axis);
        self
    }

    /// Set y_axis - See GnuplotAxis or try it out
    pub fn y_axis<'a>(&'a mut self, axis: GnuplotAxis) -> &'a mut Self
    {
        self.y_axis = Some(axis);
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
}

impl Default for GnuplotSettings{
    fn default() -> Self {
        Self{
            x_label: "".to_owned(),
            y_label: "".to_owned(),
            title: "".to_owned(),
            terminal: GnuplotTerminal::PDF,
            pallet: GnuplotPallet::PresetHSV,
            x_axis: None,
            y_axis: None,
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
    0.0 <= v && 1.0 >= v
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
    /// These are the brightness values used for calculating the palett later on.
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
    /// Will return [red, green, blue], where red, green and blue are in [0.0, 1.0],
    /// will return [0,0,0] for NAN gray value.
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

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// defines presets for different color pallets
pub enum GnuplotPallet{
    /// Use preset HSV pallet
    PresetHSV,
    /// Use preset RGB pallet
    PresetRGB,
    CubeHelix(CubeHelixParameter)
}

impl GnuplotPallet{
    pub(crate) fn write_pallet<W: Write>(&self, mut writer: W) -> std::io::Result<()>
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
                writeln!(writer, "gamma={}   # gamma < 1 emphasises low intensity values, gamma > 1 high intensity ones", helix.gamma)?;
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
            }
        }
        
    }
}


#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
/// # Options for choosing gnuplot Terminal
pub enum GnuplotTerminal{
    /// # Use EpsLatex as terminal in gnuplot
    /// * the created gnuplotscript assumes, you have `latexmk` installed
    /// * if you do not have latexmk, you can still use this, but you have to manually edit the 
    /// gnuplotskrip later on
    /// * gnuplot skript will create `.tex` file and `.pdf` file created from the tex file
    EpsLatex,
    /// # Use pdf as gnuplot terminal
    /// * gnuplot skript will create a `.pdf` file
    PDF,
}

impl GnuplotTerminal{
    pub(crate) fn terminal_str(&self) -> &'static str
    {
        match self{
            Self::EpsLatex => {
                "set t epslatex 9 standalone color size 7.4cm, 5cm header \"\\\\usepackage{amsmath}\\n\"\nset font \",9\""
            },
            Self::PDF => {
                "set t pdf"
            }
        }
    }
    
    pub(crate) fn output<W: Write>(&self, name: &str, mut writer: W) -> std::io::Result<()>
    {
        match self {
            Self::EpsLatex => {
                if name.ends_with(".tex") {
                    write!(writer, "{}", name)
                } else {
                    write!(writer, "{}.tex", name)
                }
            },
            Self::PDF => {
                if name.ends_with(".pdf") {
                    write!(writer, "{}", name)
                } else {
                    write!(writer, "{}.pdf", name)
                }
            }
        }
    }

    pub(crate) fn finish<W: Write>(&self, output_name: &str, mut w: W) -> std::io::Result<()>
    {
        match self {
            Self::EpsLatex => {
                write!(w, "system('latexmk ")?;
                self.output(output_name, &mut w)?;
                writeln!(w, "-pdf -f')")
            },
            _ => Ok(())
        }
    } 
}