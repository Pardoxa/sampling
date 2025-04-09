(function() {
    var implementors = Object.fromEntries([["sampling",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"enum\" href=\"sampling/histogram/enum.HistErrors.html\" title=\"enum sampling::histogram::HistErrors\">HistErrors</a>&gt; for <a class=\"enum\" href=\"sampling/glue/enum.GlueErrors.html\" title=\"enum sampling::glue::GlueErrors\">GlueErrors</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/heatmap/struct.CubeHelixParameter.html\" title=\"struct sampling::heatmap::CubeHelixParameter\">CubeHelixParameter</a>&gt; for <a class=\"enum\" href=\"sampling/heatmap/enum.GnuplotPalette.html\" title=\"enum sampling::heatmap::GnuplotPalette\">GnuplotPalette</a>"],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/heatmap/struct.PaletteRGB.html\" title=\"struct sampling::heatmap::PaletteRGB\">PaletteRGB</a>&gt; for <a class=\"enum\" href=\"sampling/heatmap/enum.GnuplotPalette.html\" title=\"enum sampling::heatmap::GnuplotPalette\">GnuplotPalette</a>"],["impl&lt;B, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/histogram/struct.AtomicGenericHist.html\" title=\"struct sampling::histogram::AtomicGenericHist\">AtomicGenericHist</a>&lt;B, T&gt;&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.GenericHist.html\" title=\"struct sampling::histogram::GenericHist\">GenericHist</a>&lt;B, T&gt;"],["impl&lt;B, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/histogram/struct.GenericHist.html\" title=\"struct sampling::histogram::GenericHist\">GenericHist</a>&lt;B, T&gt;&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.AtomicGenericHist.html\" title=\"struct sampling::histogram::AtomicGenericHist\">AtomicGenericHist</a>&lt;B, T&gt;"],["impl&lt;B, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;B&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.AtomicGenericHist.html\" title=\"struct sampling::histogram::AtomicGenericHist\">AtomicGenericHist</a>&lt;B, T&gt;<div class=\"where\">where\n    B: <a class=\"trait\" href=\"sampling/histogram/trait.Binning.html\" title=\"trait sampling::histogram::Binning\">Binning</a>&lt;T&gt;,</div>"],["impl&lt;B, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;B&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.GenericHist.html\" title=\"struct sampling::histogram::GenericHist\">GenericHist</a>&lt;B, T&gt;<div class=\"where\">where\n    B: <a class=\"trait\" href=\"sampling/histogram/trait.Binning.html\" title=\"trait sampling::histogram::Binning\">Binning</a>&lt;T&gt;,</div>"],["impl&lt;Ensemble, R, Hist, Energy, S, Res&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/rewl/struct.ReplicaExchangeWangLandau.html\" title=\"struct sampling::rewl::ReplicaExchangeWangLandau\">ReplicaExchangeWangLandau</a>&lt;Ensemble, R, Hist, Energy, S, Res&gt;&gt; for <a class=\"type\" href=\"sampling/rees/type.Rees.html\" title=\"type sampling::rees::Rees\">Rees</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/1.86.0/std/primitive.unit.html\">()</a>, Ensemble, R, Hist, Energy, S, Res&gt;<div class=\"where\">where\n    Hist: <a class=\"trait\" href=\"sampling/histogram/trait.Histogram.html\" title=\"trait sampling::histogram::Histogram\">Histogram</a>,</div>"],["impl&lt;HistWidth, HistHeight&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/heatmap/struct.HeatmapUsize.html\" title=\"struct sampling::heatmap::HeatmapUsize\">HeatmapUsize</a>&lt;HistWidth, HistHeight&gt;&gt; for <a class=\"struct\" href=\"sampling/heatmap/struct.HeatmapF64.html\" title=\"struct sampling::heatmap::HeatmapF64\">HeatmapF64</a>&lt;HistWidth, HistHeight&gt;<div class=\"where\">where\n    HistWidth: <a class=\"trait\" href=\"sampling/histogram/trait.Histogram.html\" title=\"trait sampling::histogram::Histogram\">Histogram</a>,\n    HistHeight: <a class=\"trait\" href=\"sampling/histogram/trait.Histogram.html\" title=\"trait sampling::histogram::Histogram\">Histogram</a>,</div>"],["impl&lt;HistWidth, HistHeight&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/heatmap/struct.HeatmapUsizeMean.html\" title=\"struct sampling::heatmap::HeatmapUsizeMean\">HeatmapUsizeMean</a>&lt;HistWidth, HistHeight&gt;&gt; for <a class=\"type\" href=\"sampling/heatmap/type.HeatmapU.html\" title=\"type sampling::heatmap::HeatmapU\">HeatmapU</a>&lt;HistWidth, HistHeight&gt;"],["impl&lt;R, Hist, Energy, S, Res&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/rewl/struct.RewlWalker.html\" title=\"struct sampling::rewl::RewlWalker\">RewlWalker</a>&lt;R, Hist, Energy, S, Res&gt;&gt; for <a class=\"struct\" href=\"sampling/rees/struct.ReesWalker.html\" title=\"struct sampling::rees::ReesWalker\">ReesWalker</a>&lt;R, Hist, Energy, S, Res&gt;<div class=\"where\">where\n    Hist: <a class=\"trait\" href=\"sampling/histogram/trait.Histogram.html\" title=\"trait sampling::histogram::Histogram\">Histogram</a>,</div>"],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/histogram/struct.AtomicHistogramFloat.html\" title=\"struct sampling::histogram::AtomicHistogramFloat\">AtomicHistogramFloat</a>&lt;T&gt;&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.HistogramFloat.html\" title=\"struct sampling::histogram::HistogramFloat\">HistogramFloat</a>&lt;T&gt;"],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.86.0/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"sampling/histogram/struct.HistogramFloat.html\" title=\"struct sampling::histogram::HistogramFloat\">HistogramFloat</a>&lt;T&gt;&gt; for <a class=\"struct\" href=\"sampling/histogram/struct.AtomicHistogramFloat.html\" title=\"struct sampling::histogram::AtomicHistogramFloat\">AtomicHistogramFloat</a>&lt;T&gt;"]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[7082]}