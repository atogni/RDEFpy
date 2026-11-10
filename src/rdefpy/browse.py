# Copyright (c) 2026 Morehead State University
# Author: Andrea Togni
# SPDX-License-Identifier: MIT

"""Interactive browser: spectrum, carrier and headers around a chosen epoch."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from rdefpy.carrier import CARRIER_DTYPE, extract_carrier

_MAX_POINTS = 4096  # spectrum bins drawn; more are max-pooled (display only)
_ONE_S = np.timedelta64(1_000_000_000, "ns")
_CACHE_BYTES = 1 << 30  # decoded records, spectra, carrier blocks; least recently used freed first above this


def _chebwin(m: int, at: float = 100.0) -> np.ndarray:
    """Dolph-Chebyshev window, sidelobes *at* dB down (frequency-sampling design)."""
    if m < 3:
        return np.ones(m)
    order = m - 1
    beta = np.cosh(np.arccosh(10 ** (at / 20)) / order)
    x = beta * np.cos(np.pi * np.arange(m) / m)
    p = np.where(np.abs(x) <= 1, np.cos(order * np.arccos(np.clip(x, -1, 1))),
                 np.sign(x) ** order * np.cosh(order * np.arccosh(np.maximum(np.abs(x), 1))))
    if m % 2:
        w = np.real(np.fft.fft(p))[: (m + 1) // 2]
        w = np.concatenate((w[:0:-1], w))
    else:
        w = np.real(np.fft.fft(p * np.exp(1j * np.pi * np.arange(m) / m)))[: m // 2 + 1]
        w = np.concatenate((w[:0:-1], w[1:]))
    return w / w.max()


# Crameri batlow (colour-blind friendly, perceptually uniform), every 15th of its 256
# entries (cmcrameri batlow.txt); linear between them, max RGB error 0.008.
# Copyright (c) 2020 Fabio Crameri, MIT License (same terms as this package's LICENSE).
# Crameri, F., Scientific colour maps, Zenodo, doi:10.5281/zenodo.1243862.
_BATLOW = [(0.005, 0.098, 0.350), (0.048, 0.186, 0.365), (0.065, 0.255, 0.376), (0.083, 0.311, 0.384),
           (0.119, 0.363, 0.383), (0.183, 0.406, 0.362), (0.266, 0.439, 0.320), (0.358, 0.467, 0.269),
           (0.456, 0.496, 0.218), (0.563, 0.524, 0.177), (0.678, 0.551, 0.178), (0.791, 0.571, 0.236),
           (0.890, 0.592, 0.330), (0.962, 0.624, 0.456), (0.990, 0.667, 0.593), (0.993, 0.711, 0.721),
           (0.989, 0.754, 0.847), (0.981, 0.800, 0.981)]

_WINDOWS = {  # applied to the spectrum only, on request; the carrier uses the samples as recorded
    "none": np.ones,
    "Hann": np.hanning,
    "Hamming": np.hamming,
    "Blackman": np.blackman,
    "Kaiser 8": lambda m: np.kaiser(m, 8.0),
    "Chebyshev 100 dB": _chebwin,
}


def _fmt(t) -> str:
    """ISO epoch without trailing zero digits: 2026-02-08T12:00:00.5."""
    s = str(np.datetime64(t, "ns"))
    return s.rstrip("0").rstrip(".") if "." in s else s


_BAD_INPUT = ("Epoch: ISO; T: s > 0; blocks: int >= 0; zero pad: int 1 to 10; "
              "Periodogram, Spectrogram: numbers, min < max")
_BAR_PX = 10  # header scrollbar width
_HDR_PT = 8.0  # header font size, smaller only if the longest line would not fit


def _text_image(text: str, width_px: float, dpi: float, pt: float) -> np.ndarray:
    """White monospace *text* on black as an RGBA image, *width_px* wide."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    f = Figure(figsize=(width_px / dpi, (text.count("\n") + 2) * pt * 1.5 / 72), dpi=dpi, facecolor="black")
    f.text(0, 1, text, va="top", family="monospace", fontsize=pt, color="white")
    FigureCanvasAgg(f).draw()
    img = np.asarray(f.canvas.buffer_rgba())
    rows = np.flatnonzero(img[..., :3].max(axis=(1, 2)) > 40)  # rows with ink
    return img[: rows[-1] + round(pt * dpi / 72) if len(rows) else 1]  # to the last line, plus a margin


def _text_width(line: str, dpi: float, pt: float) -> float:
    """Pixel width of one monospace *line* as Agg draws it (hinting rounds each advance to whole pixels)."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    return RendererAgg(1, 1, dpi).get_text_width_height_descent(
        line, FontProperties(family="monospace", size=pt), ismath=False)[0]


def _limits(lo: str, hi: str):
    """Two cells -> (lo, hi), None where blank; ValueError unless finite numbers with lo < hi."""
    lo, hi = (float(s) if s.strip() else None for s in (lo, hi))
    if any(v is not None and not np.isfinite(v) for v in (lo, hi)) or (
            lo is not None and hi is not None and not lo < hi):
        raise ValueError
    return lo, hi


def browse(pf, epoch=None, integration_time: float = 1.0, blocks: int = 5):
    """
    Open a dark matplotlib window to inspect a product file around an epoch.

    Boxes at the bottom: centre epoch (time scale of the record time tags,
    e.g. ``2026-02-08T12:00:00.5``), integration time T [s] (sub-second or
    longer than a record; ``T * sample_rate`` must be whole), carrier blocks
    on each side, zero-padding factor (integer 1 to 10), spectrum window, and two groups of
    frequency min / max [Hz] and power min / max [dB] cells: "Periodogram"
    (spectrum axes; blank: +- Nyquist, auto) and "Spectrogram" (its
    frequency axis and colour range; blank: +- Nyquist, median to max dB).
    A checkbox shows or hides the carrier estimates on the spectrogram. The arrow
    buttons (or the left and right arrow keys) move the epoch by T. While
    typing, Tab / Shift+Tab apply the cell and move to the next / previous one.
    The values under the mouse show above the boxes; the mouse wheel or the
    scrollbar scrolls the headers. For the
    window [epoch - T/2, epoch + T/2):

    - left: power spectrum of the samples in dB, frequency from the channel
      centre. The window applies to the spectrum and spectrogram only. Power is
      ``abs(FFT)^2 / sum(w)^2``, so a tone of amplitude A peaks near
      ``20 log10 A``. Spectra longer than 4096 bins are max-pooled for
      drawing only (title says so);
    - middle: ``extract_carrier`` (samples as recorded, never windowed;
      ``zero_padding`` = the zero-pad box) on the window and up
      to *blocks* blocks of the same T before and after it: the signal
      frequency before downconversion and the variable part of the
      downconversion frequency, both minus its fixed part f_DC,fixed
      (§5.2.6), and the baseband frequency (after downconversion) drawn
      over a spectrogram of the same blocks (same T, zero padding and
      window as the spectrum; frequency and colour range = the Spectrogram
      cells). The window is shaded or dotted;
    - right: the header of the first record the window touches, then for
      each further record the header lines that differ from it.

    When the window is not inside the file, crosses a gap or crosses a
    sample-rate change, nothing new is drawn and a red title on the
    spectrum says why. Records are read only when a window touches them; decoded
    records, spectra and carrier blocks are kept for revisits, the least
    recently used freed first beyond 1 GiB. Needs matplotlib (``pip install "rdefpy[browse]"``); returns the figure.

    :param pf: ``ProductFile`` from ``read_product``.
    :param epoch: initial centre epoch; default: the middle of the first
        record.
    :param integration_time: initial T in seconds.
    :param blocks: initial number of carrier blocks on each side.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Bbox, IdentityTransform
        from matplotlib.patheffects import Normal, Stroke
        from matplotlib.widgets import Button, CheckButtons, TextBox
    except ImportError as e:
        raise ImportError("browse needs matplotlib: pip install \"rdefpy[browse]\" or pip install matplotlib") from e
    class TextBox(TextBox):  # matplotlib's redraws the whole figure on every click, key and hover
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cursor.set_color("white")  # base: black, invisible on the dark boxes

        def stop_typing(self):
            if self.capturekeystrokes:  # base: full draw per box on any click anywhere
                super().stop_typing()

        def _rendercursor(self):  # repaint only this box instead of the figure
            canvas = self.ax.figure.canvas  # a subfigure's canvas is its root figure's
            if not canvas.supports_blit or canvas.figure._get_renderer() is None:
                return super()._rendercursor()
            canvas.draw = lambda: (canvas.figure.draw_artist(self.ax), canvas.blit(self.ax.bbox))
            try:
                super()._rendercursor()
            finally:
                del canvas.draw  # back to the class method

    if len(pf) == 0:
        raise ValueError("product file has no records")
    first, last = pf[0].header.timetag, pf[-1].header.timetag + _ONE_S
    if epoch is None:
        epoch = first + _ONE_S // 2

    batlow = LinearSegmentedColormap.from_list("batlow", _BATLOW)
    style = {"axes.grid": True, "grid.alpha": 0.3, "axes.formatter.useoffset": False,
             "axes.formatter.limits": (-7, 12)}  # ticks show the values: no offset, no 1e5
    dark = lambda: plt.style.context(["dark_background", style])  # every redraw
    box = dict(color="0.15", hovercolor="0.25")
    tbox = dict(color="0.15", hovercolor="0.15")  # no hover change: it would redraw the figure
    win = {"name": "none"}
    with dark():
        fig = plt.figure(figsize=(17, 8.5))
        ax = fig.add_axes((0.05, 0.31, 0.42, 0.62))
        ax_sky = fig.add_axes((0.53, 0.66, 0.135, 0.27))  # same x extent as ax_res: one time axis
        ax_res = fig.add_axes((0.53, 0.33, 0.135, 0.25), sharex=ax_sky)
        ax_txt = fig.add_axes((0.71, 0.03, 0.29, 0.94))
        sm = ScalarMappable(cmap=batlow)  # spectrogram colour range; its colorbar is made once, updated per redraw
        cbar = fig.colorbar(sm, cax=fig.add_axes((0.67, 0.33, 0.007, 0.25)))
        cbar.set_label("Power [dB]", fontsize=9)
        cbar.ax.tick_params(colors="white", labelsize=8)
        row0, row1, row2 = 0.17, 0.10, 0.03
        tb_epoch = TextBox(fig.add_axes((0.08, row1, 0.17, 0.04)), "Epoch ", initial=_fmt(epoch), **tbox)
        b_prev = Button(fig.add_axes((0.26, row1, 0.025, 0.04)), "◀", **box)
        b_next = Button(fig.add_axes((0.29, row1, 0.025, 0.04)), "▶", **box)
        tb_T = TextBox(fig.add_axes((0.36, row1, 0.05, 0.04)), "T [s] ", initial=f"{integration_time:g}", **tbox)
        tb_n = TextBox(fig.add_axes((0.53, row1, 0.03, 0.04)), "Blocks each side ", initial=str(blocks), **tbox)
        tb_pad = TextBox(fig.add_axes((0.08, row2, 0.03, 0.04)), "Zero pad x ", initial="1", **tbox)
        b_win = Button(fig.add_axes((0.13, row2, 0.11, 0.04)), "Window: none ▾", **box)
        opts = [Button(fig.add_axes((0.13, row2 + 0.04 + 0.035 * i, 0.11, 0.035)), name, color="0.25",
                       hovercolor="0.4") for i, name in enumerate(_WINDOWS)]  # drop-down, hidden until opened
        cells = []  # f min, f max, P min, P max of the Periodogram (row 2), then of the Spectrogram (row 0)
        for row, group in ((row2, "Periodogram"), (row0, "Spectrogram")):
            fig.text(0.255, row + 0.02, group, va="center", fontweight="bold")
            cells += [TextBox(fig.add_axes((x, row, 0.04, 0.04)), label, initial="", **tbox)
                      for x, label in ((0.375, "f min [Hz] "), (0.47, "f max [Hz] "), (0.565, "P min [dB] "),
                                       (0.66, "P max [dB] "))]
        tb_fmin, tb_fmax, tb_pmin, tb_pmax, tb_sfmin, tb_sfmax, tb_spmin, tb_spmax = cells
        cb = CheckButtons(fig.add_axes((0.57, row1, 0.13, 0.04)), ["Carrier on spectrogram"], [True],
                          label_props=dict(fontsize=[9]), frame_props=dict(edgecolor="white", facecolor="0.15", s=[80]),
                          check_props=dict(facecolor="white", s=[80]))
        cb.ax.set_facecolor("0.15")
    boxes = (tb_epoch, tb_T, tb_n, tb_pad, *cells)
    overlay = []  # carrier estimates drawn over the spectrogram; the checkbox shows / hides them

    def message(a, text):
        a.set_title(text, color="tomato", fontsize=9)
        fig.canvas.draw_idle()

    def parse():
        mid = np.datetime64(tb_epoch.text.strip().rstrip("Z"), "ns")
        T, n_side, pad = float(tb_T.text), int(tb_n.text), int(tb_pad.text)
        # NaT, inf and counts beyond int64 ns would raise or wrap in the time arithmetic
        if np.isnat(mid) or not 0 < T * 1e9 < 2**63 or not 0 <= n_side < 2**63 or not 1 <= pad <= 10:
            raise ValueError
        return (mid, T, n_side, pad) + tuple(_limits(lo.text, hi.text) for lo, hi in zip(cells[::2], cells[1::2]))

    def clear():
        for a in (ax, ax_sky, ax_res, ax_txt):
            a.clear()
            # ticks are made lazily at draw time, outside this style: fix their colours here
            a.tick_params(which="both", colors="white", grid_color="white", grid_alpha=0.3)
        ax_txt.axis("off")
        cbar.ax.set_visible(False)  # shown again with a spectrogram
        if hdr["im"] is not None:
            hdr["im"].remove()
            hdr["im"] = None

    def show(_=None):
        with dark():
            clear()
            _show()

    cache, used = OrderedDict(), [0]
    hdr = {"top": 0, "shown": None, "img": None, "im": None, "drag": False}  # header pixels: wanted and shown first row
    px = IdentityTransform()  # display pixels
    track = fig.add_artist(Rectangle((0, 0), 0, 0, transform=px, color="0.2", visible=False))  # scrollbar
    thumb = fig.add_artist(Rectangle((0, 0), 0, 0, transform=px, color="0.6", visible=False))

    def hdr_view():
        """Put rows top.. of the header pixels in the panel, 1:1 (no resampling); True if moved."""
        b, img = ax_txt.bbox, hdr["img"]
        top = min(max(int(hdr["top"]), 0), max(len(img) - int(b.height), 0))
        moved, hdr["top"], hdr["shown"] = top != hdr["shown"], top, top
        part = img[top : top + int(b.height)]
        if hdr["im"] is None:
            hdr["im"] = fig.figimage(part, origin="upper")
        hdr["im"].set_data(part)
        hdr["im"].ox, hdr["im"].oy = b.x0, b.y1 - len(part)
        over = len(img) > b.height
        track.set_bounds(b.x1 - _BAR_PX, b.y0, _BAR_PX - 2, b.height)
        thumb.set_bounds(b.x1 - _BAR_PX, b.y1 - (top + b.height) * b.height / len(img), _BAR_PX - 2,
                         b.height ** 2 / len(img))
        track.set_visible(over)
        thumb.set_visible(over)
        return moved

    def cached(key, make):
        """make() once per key; least recently used entries freed above _CACHE_BYTES."""
        if key in cache:
            cache.move_to_end(key)
            return cache[key][0]
        v = make()
        n = sum(x.nbytes for x in (v if isinstance(v, tuple) else (v,)) if isinstance(x, np.ndarray))
        cache[key] = v, n
        used[0] += n
        while used[0] > _CACHE_BYTES and len(cache) > 1:
            used[0] -= cache.popitem(last=False)[1][1]
        return v

    def touched(a, T):
        """Records holding [a, a + T), their tags and sample rate; ValueError if not drawable."""
        b = a + np.timedelta64(round(T * 1e9), "ns")
        if a < first or b > last:
            raise ValueError(f"Window outside the file ({_fmt(first)} to {_fmt(last)})")
        sub = pf.select(start_epoch=a - _ONE_S + np.timedelta64(1, "ns"), stop_epoch=b)
        tags = [r.header.timetag for r in sub]
        if (len(sub) == 0 or tags[0] > a or tags[-1] + _ONE_S < b  # a record missing at either end
                or any(t1 - t0 != _ONE_S for t0, t1 in zip(tags, tags[1:]))):
            raise ValueError("Window crosses a gap between records")
        fs = float(sub[0].header.sample_rate)
        if any(float(r.header.sample_rate) != fs for r in sub):
            raise ValueError("Window crosses a sample-rate change")
        if abs(T * fs - round(T * fs)) > 1e-9 * T * fs:
            raise ValueError(f"T * sample_rate must be whole ({fs:g} Hz)")
        return sub, tags, fs

    def samples(a, T):
        """The samples in [a, a + T), as recorded, and the sample rate."""
        sub, tags, fs = touched(a, T)
        i0 = round((a - tags[0]) / np.timedelta64(1, "ns") * fs / 1e9)
        recs = [cached(("rec", t), r.read_samples) for r, t in zip(sub, tags)]  # each record decoded once
        return np.concatenate(recs)[i0 : i0 + round(T * fs)], fs

    def spectrum(a, T, pad):
        """dB spectrum of the samples in [a, a + T), max-pooled to _MAX_POINTS: (dB, f, k, n, fs)."""
        def make():
            x, fs = samples(a, T)
            w = _WINDOWS[win["name"]](len(x))  # display only
            p = np.fft.fftshift(np.abs(np.fft.fft(x * w, pad * len(x))) ** 2) / np.sum(w) ** 2
            f = np.fft.fftshift(np.fft.fftfreq(len(p), 1 / fs))
            k = -(-len(p) // _MAX_POINTS)
            if k > 1:  # display only: max of each group of k bins
                m = len(p) // k * k
                pg, fg, r = p[:m].reshape(-1, k), f[:m].reshape(-1, k), np.arange(m // k)
                i = pg.argmax(1)  # each group's peak bin, at its own frequency
                p, f = pg[r, i], fg[r, i]
            return 10 * np.log10(np.maximum(p, np.finfo(float).tiny)), f, k, len(x), fs
        return cached(("spec", a, T, pad, win["name"]), make)

    def carrier(a, T, pad):
        """extract_carrier on [a, a + T): samples as recorded, never windowed; zero padding = the box."""
        def make():
            try:
                sub = touched(a, T)[0]
                return extract_carrier(sub, T, zero_padding=pad, start_epoch=a)[:1]
            except ValueError:  # gap or rate change: NaN row
                row = np.full(1, np.nan, dtype=CARRIER_DTYPE)
                row["time"] = a + np.timedelta64(round(T * 5e8), "ns")
                return row
        return cached(("car", a, T, pad), make)

    def _show():
        try:
            mid, T, n_side, pad, (fmin, fmax), (pmin, pmax), (sfmin, sfmax), (spmin, spmax) = parse()
        except ValueError:
            return message(ax, _BAD_INPUT)
        half = np.timedelta64(round(T * 5e8), "ns")
        a, b = mid - half, mid + half
        try:
            sub, tags, fs = touched(a, T)
            db, f, k, n, fs = spectrum(a, T, pad)
        except ValueError as e:
            return message(ax, str(e))
        title = f"{_fmt(mid)}   T = {T:g} s   ({n} samples)"
        if k > 1:
            title += f"   max of every {k} bins"
        ax.plot(f, db, lw=0.7, color="deepskyblue")
        ax.set(title=title, xlabel="Frequency from channel centre [Hz]", ylabel="Power [dB]")
        lims = [(-fs / 2 if fmin is None else fmin, fs / 2 if fmax is None else fmax)]  # blank: automatic; set at the end
        cw = carrier(a, T, pad)[0]  # this window's extract_carrier row
        if np.isfinite(cw["frequency"]):  # transparent dot at the carrier, with its values
            def amp():  # |mean of the samples dechirped at the carrier|: unwindowed, same scale as the spectrum
                x, fs_ = samples(a, T)
                u = np.arange(len(x)) / fs_ - T / 2  # from the block midpoint, where frequency is tagged
                return np.abs([np.mean(x * np.exp(-2j * np.pi * (cw["frequency"] * u + cw["rate"] / 2 * u**2)))])
            pc = 20 * np.log10(max(cached(("amp", a, T, pad), amp)[0], np.finfo(float).tiny))
            ax.plot(cw["frequency"], pc, "o", ms=12, color="white", alpha=0.35)
            ax.annotate(f"carrier (extract_carrier)\n{cw['frequency']:.4f} Hz\n{pc:.1f} dB", (cw["frequency"], pc), xytext=(12, -10), va="top",
                        textcoords="offset points", color="white", fontsize=9,
                        bbox=dict(boxstyle="round", fc="black", ec="0.5", alpha=0.6))
        y0, y1 = ax.get_ylim()  # after the dot: a blank side autoscales over it too
        lims.append((y0 if pmin is None else pmin, y1 if pmax is None else pmax))

        hl = lambda h: [ln[2:] if ln.startswith("  ") else ln for ln in str(h).splitlines()]  # fits the panel
        h0 = hl(sub[0].header)
        lines = [f"Record {_fmt(tags[0])}"] + h0
        for r, tag in zip(list(sub)[1:], tags[1:]):
            lines += ["", f"Record {_fmt(tag)}, changed:"]
            lines += [ln for ln in hl(r.header) if ln not in h0]
        text = "\n".join(lines)
        ax_txt.text(0, 1, text, visible=False)  # the full text, for programs
        w = ax_txt.bbox.width - _BAR_PX - 4  # room for the scrollbar
        hdr["pt"] = min(_HDR_PT, w * 72 / (fig.dpi * 0.64 * max(map(len, lines))))  # mono: ~0.64 em/char
        while hdr["pt"] > 4 and _text_width(max(lines, key=len), fig.dpi, hdr["pt"]) > w:  # hinted: whole px/char
            hdr["pt"] -= 0.1
        hdr["img"] = _text_image(text, w, fig.dpi, hdr["pt"])  # drawn once: scrolling moves pixels only
        hdr_view()

        # Carrier: the window and n_side blocks of T each side, as far as the file goes.
        Tns = np.timedelta64(round(T * 1e9), "ns")
        start = a - min(n_side, int((a - first) / Tns)) * Tns
        stop = b + min(n_side, (last - b) // Tns) * Tns
        starts = [start + i * Tns for i in range((stop - start) // Tns)]
        c = np.concatenate([carrier(s, T, pad) for s in starts])
        t = (c["time"] - mid) / np.timedelta64(1, "ns") / 1e9
        h = sub[0].header
        fixed = float(h.rf_to_if_downconv) + float(h.if_to_channel_downconv)  # f_DC,fixed (§5.2.6.1)
        ax_sky.plot(t, c["frequency_before_downconversion"] - fixed, "o-", ms=3, color="gold", label="before downconversion")
        ax_sky.plot(t, c["downconversion_frequency"] - fixed, "s--", ms=3, color="violet", label="f_DC,variable")
        ax_sky.set_title(f"Minus f_DC,fixed = {fixed:,.0f} Hz", fontsize=9)
        ax_sky.set_ylabel("Hz")
        ax_sky.legend(fontsize=7)
        # Spectrogram of the same blocks (same T, zero pad, window), carrier on top.
        cols = []
        for s in starts:
            try:
                cols.append(spectrum(s, T, pad)[0])
            except ValueError:  # block over a gap: blank column
                cols.append(np.full(len(f), np.nan))
        sg = np.array(cols).T
        df = f[1] - f[0]
        lims += [(-fs / 2 if sfmin is None else sfmin, fs / 2 if sfmax is None else sfmax),
                 (np.nanmedian(sg) if spmin is None else spmin, np.nanmax(sg) if spmax is None else spmax)]
        if any(not lo < hi for lo, hi in lims):  # one cell set beyond the automatic other side
            clear()
            return message(ax, _BAD_INPUT)
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax_res.imshow(sg, origin="lower", aspect="auto", cmap=batlow, interpolation="nearest",
                      extent=(t[0] - T / 2, t[-1] + T / 2, f[0], f[-1] + df), clim=lims[3])
        sm.set_clim(lims[3])  # the colorbar follows
        cbar.ax.set_visible(True)
        # black on the pale-pink ridge (batlow max), white halo on the dark-blue floor (batlow min)
        # marker: frequency at the block midpoint; dashes either side, clear of it: the FrFT rate
        halo = [Stroke(linewidth=3, foreground="white"), Normal()]
        dt = np.array([-0.42, -0.12, np.nan, 0.12, 0.42, np.nan]) * T
        overlay[:] = (ax_res.plot(t, c["frequency"], "o", ms=4, color="black", path_effects=halo)
                      + ax_res.plot((t[:, None] + dt).ravel(), (c["frequency"][:, None] + c["rate"][:, None] * dt).ravel(),
                                    lw=1.2, color="black", path_effects=halo))
        for ln in overlay:
            ln.set_visible(cb.get_status()[0])
        ax_res.set_ylim(lims[2])
        ax_res.set_title("Baseband (after downconversion)", fontsize=9)
        ax_res.set(xlabel="s from epoch", ylabel="Hz")
        ax_sky.axvspan(-T / 2, T / 2, color="0.3")
        for x_ in (-T / 2, T / 2):
            ax_res.axvline(x_, color="0.7", lw=0.6, ls=":")
        fig.canvas.draw_idle()

    def shift(step):
        try:
            mid, T = parse()[:2]
        except ValueError:
            return
        tb_epoch.set_val(_fmt(mid + step * np.timedelta64(round(T * 1e9), "ns")))  # submits: redraws
        tb_epoch.cursor.set_visible(tb_epoch.capturekeystrokes)  # set_val shows a cursor

    def menu(open_):
        for b in opts:
            b.ax.set_visible(open_)
            b.set_active(open_)
        tb_epoch.set_active(not open_)  # the open menu covers it
        fig.canvas.draw_idle()

    def pick(name):
        win["name"] = name
        b_win.label.set_text(f"Window: {name} ▾")
        menu(False)
        show()

    tab_order = (tb_epoch, tb_T, tb_n, tb_pad, tb_sfmin, tb_sfmax, tb_spmin, tb_spmax,
                 tb_fmin, tb_fmax, tb_pmin, tb_pmax)

    def tab(event):
        """Tab / Shift+Tab while typing: apply this cell (as Enter does), type in the next / previous one."""
        i = next((i for i, tb in enumerate(tab_order) if tb.capturekeystrokes), None)
        if i is None:  # no cell being typed in: nothing to do
            return
        # Tk passes Shift+Tab as "tab" with the Shift bit in the state; X11 / Qt name it *iso_left_tab / *backtab
        state = getattr(event.guiEvent, "state", 0)
        back = event.key != "tab" or (isinstance(state, int) and state & 1)
        step = -1 if back else 1
        k = (i + step) % len(tab_order)
        while not tab_order[k].active:  # the epoch box under the open window menu
            k = (k + step) % len(tab_order)
        canvas = fig.canvas
        canvas.draw = lambda: None  # stop_typing's full draw: the submit's redraw (draw_idle) repaints
        try:
            tab_order[i].stop_typing()  # submits: show()
        finally:
            del canvas.draw  # back to the class method
        nxt = tab_order[k]
        nxt.begin_typing()
        nxt.cursor_index = len(nxt.text)
        nxt._rendercursor()

    def on_key(event):
        if event.key and event.key.endswith("tab"):  # tab, shift+tab, shift+iso_left_tab, backtab
            return tab(event)
        step = {"right": 1, "left": -1}.get(event.key)
        if step and not any(tb.capturekeystrokes for tb in boxes):
            shift(step)

    for tb in boxes:
        tb.on_submit(show)
    b_prev.on_clicked(lambda _: shift(-1))
    b_next.on_clicked(lambda _: shift(1))
    b_win.on_clicked(lambda _: menu(not opts[0].active))
    for b in opts:
        b.on_clicked(lambda _, name=b.label.get_text(): pick(name))

    def toggle(_):  # no recompute: the overlay is only hidden; redraws read the box
        for ln in overlay:
            ln.set_visible(cb.get_status()[0])
        fig.canvas.draw_idle()

    cb.on_clicked(toggle)
    menu(False)
    fig.canvas.mpl_connect("key_press_event", on_key)
    if hasattr(fig.canvas, "get_tk_widget"):  # Tk's default Tab moves keyboard focus off the canvas: stop it
        for seq in ("<Tab>", "<ISO_Left_Tab>"):
            try:
                fig.canvas.get_tk_widget().bind(seq, lambda e: fig.canvas.key_press(e) or "break")
            except Exception:  # keysym unknown to this Tk
                pass

    # Values under the mouse, repainted alone (blit): no full redraw per mouse move.
    readout, bg = fig.text(0.05, 0.225, "", fontsize=9, color="0.85", animated=True), [None]
    strip = lambda: Bbox.from_extents(0.04, 0.215, 0.6, 0.25).transformed(fig.transFigure)  # under readout

    def on_draw(_):
        bg[0] = fig.canvas.copy_from_bbox(strip()) if fig.canvas.supports_blit else None
        fig.draw_artist(readout)

    def on_move(e):
        x, y = e.xdata, e.ydata
        if e.inaxes is ax:
            text = f"f = {x:.3f} Hz   P = {y:.1f} dB"
        elif e.inaxes is ax_sky:
            text = f"t = {x:+.3f} s from epoch   f = {y:.4f} Hz + f_DC,fixed"
        elif e.inaxes is ax_res:
            v = ax_res.images[0].get_cursor_data(e) if ax_res.images else None
            text = f"t = {x:+.3f} s from epoch   f = {y:.3f} Hz" + (f"   P = {v:.1f} dB" if v is not None else "")
        else:
            text = ""
        if text == readout.get_text():
            return
        readout.set_text(text)
        if bg[0] is not None:
            fig.canvas.restore_region(bg[0])
            fig.draw_artist(readout)
            fig.canvas.blit(strip())

    fig.canvas.mpl_connect("draw_event", on_draw)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    # Mouse wheel over the headers scrolls them; only that panel is repainted.
    cover = fig.add_artist(Rectangle((0, 0), 1, 1, transform=ax_txt.transAxes, color="black", animated=True))

    def on_scroll(e):
        if e.inaxes is not ax_txt or hdr["im"] is None:
            return
        hdr["top"] -= round(e.step * 3 * hdr["pt"] * 1.2 * fig.dpi / 72)  # 3 lines per wheel step
        repaint()

    def repaint():
        if not hdr_view():
            return
        if bg[0] is None:
            return fig.canvas.draw_idle()
        for a_ in (cover, hdr["im"], track, thumb):  # cover paints over the old view
            fig.draw_artist(a_)
        fig.canvas.blit(ax_txt.bbox)

    def on_bar(e):  # press on the scrollbar, or drag with the button held: thumb centre to the mouse
        b = ax_txt.bbox
        if e.name == "button_release_event":
            hdr["drag"] = False
            return
        if e.name == "button_press_event":
            hdr["drag"] = track.get_visible() and b.x1 - _BAR_PX <= e.x <= b.x1 and b.y0 <= e.y <= b.y1
        if hdr["drag"]:
            hdr["top"] = round((b.y1 - e.y) / b.height * len(hdr["img"]) - b.height / 2)
            repaint()

    for name in ("button_press_event", "motion_notify_event", "button_release_event"):
        fig.canvas.mpl_connect(name, on_bar)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("resize_event", show)  # header pixels are sized to the panel
    # keep alive. Order: epoch, T, blocks, prev, next, zero pad, window, Periodogram f min, f max, P min, P max,
    # Spectrogram f min, f max, P min, P max, carrier checkbox, then the window options (index 16 on)
    fig._rdefpy_widgets = (tb_epoch, tb_T, tb_n, b_prev, b_next, tb_pad, b_win, *cells, cb, *opts)
    show()
    win_ = getattr(fig.canvas.manager, "window", None)  # fill the screen: a 17-inch figure is wider than most
    for maximize in (lambda: win_.state("zoomed"), lambda: win_.showMaximized()):  # Tk (Windows), Qt
        try:
            maximize()
            break
        except Exception:  # other backends or platforms: keep the default size
            pass
    plt.show()
    return fig
