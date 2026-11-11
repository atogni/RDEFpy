"""browse: spectrum, carrier and headers around a typed epoch (headless backend)."""

import sys
from pathlib import Path

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")
plt.switch_backend("Agg")

import rdef_spec as spec  # noqa: E402
from rdefpy import browse, extract_carrier, read_product  # noqa: E402
from rdefpy.browse import _chebwin, _fmt  # noqa: E402

REF = Path(__file__).parent / "reference" / "sample_16bit.prd"  # 2 records, 2 kHz


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_browse_window_blocks_and_buttons():
    pf = read_product(REF)
    t0 = pf[0].header.timetag
    fig = browse(pf)  # middle of the first record, T = 1 s
    ax, ax_sky, ax_res, ax_txt = fig.axes[:4]
    (tb_epoch, tb_T, tb_n, b_prev, b_next, tb_pad, b_win, tb_fmin, tb_fmax, tb_pmin, tb_pmax,
     tb_sfmin, tb_sfmax, tb_spmin, tb_spmax, cb, *opts) = fig._rdefpy_widgets
    assert tb_epoch.text == _fmt(t0).split(".")[0] + ".5"  # no trailing zeros
    assert "T = 1 s" in ax.get_title() and len(ax.lines) == 2  # spectrum + peak dot
    c = extract_carrier(pf, 1.0, zero_padding=1)[0]  # the window = the first 1-s block
    x = pf.samples[:2000]
    u = np.arange(2000) / 2000 - 0.5
    pc = 20 * np.log10(abs(np.mean(x * np.exp(-2j * np.pi * (c["frequency"] * u + c["rate"] / 2 * u**2)))))
    np.testing.assert_allclose(ax.lines[1].get_xydata(), [[c["frequency"], pc]])  # dot on the carrier
    assert ax.texts[0].get_text() == f"carrier (extract_carrier)\n{c['frequency']:.4f} Hz\n{pc:.1f} dB"
    assert "changed" not in ax_txt.texts[0].get_text()
    assert ax.get_xlim() == (-1000, 1000)  # +- Nyquist

    tb_pad.set_val("4")
    assert "max of every 2 bins" in ax.get_title()  # 8000 bins drawn as 4000
    assert all(a.xaxis.get_gridlines()[0].get_visible() for a in (ax, ax_sky, ax_res))  # grids on
    assert not any(a.yaxis.get_major_formatter().get_useOffset() for a in (ax, ax_sky, ax_res))  # true tick values
    assert not any(b.ax.get_visible() for b in opts)  # drop-down closed
    assert all((tb.cursor.get_color() == 1).all() for tb in (tb_epoch, tb_T, tb_pmax))  # white cursor
    b_win._observers.process("clicked", None)
    assert all(b.ax.get_visible() and b.active for b in opts) and not tb_epoch.active
    next(b for b in opts if b.label.get_text() == "Hann")._observers.process("clicked", None)
    assert b_win.label.get_text() == "Window: Hann ▾" and not opts[0].ax.get_visible() and tb_epoch.active
    tb_fmin.set_val("-50")
    tb_fmax.set_val("50")
    tb_pmin.set_val("0")
    tb_pmax.set_val("100")
    assert ax.get_xlim() == (-50, 50) and ax.get_ylim() == (0, 100)
    sg = np.asarray(ax_res.images[0].get_array())  # spectrogram: own cells, blank: median to max dB
    assert ax_res.get_ylim() == (-1000, 1000) and ax_res.images[0].get_clim() == (np.nanmedian(sg), np.nanmax(sg))
    tb_sfmin.set_val("-20")
    tb_sfmax.set_val("20")
    tb_spmin.set_val("10")
    tb_spmax.set_val("90")
    assert ax_res.get_ylim() == (-20, 20) and ax_res.images[0].get_clim() == (10, 90)
    assert ax.get_xlim() == (-50, 50) and ax.get_ylim() == (0, 100)  # spectrum unchanged
    cax, n_axes = fig.axes[4], len(fig.axes)  # spectrogram colorbar: made once
    assert cax.get_ylim() == (10, 90) and cax.get_visible() and cax.get_ylabel() == "Power [dB]"
    p_sky, p_res, p_cax = ax_sky.get_position(), ax_res.get_position(), cax.get_position()
    assert (p_sky.x0, p_sky.x1) == (p_res.x0, p_res.x1) and p_cax.x0 > p_res.x1  # one time axis, bar beside
    fig.canvas.draw()
    assert cax.get_tightbbox().x1 < ax_txt.bbox.x0  # ticks and label clear of the header panel

    tb_T.set_val("0.25")
    tb_n.set_val("2")
    tb_epoch.set_val(_fmt(t0 + np.timedelta64(1, "s")))  # straddles records 0 and 1
    assert "changed" in ax_txt.texts[0].get_text()
    np.testing.assert_allclose(ax_res.lines[0].get_xdata(), [-0.5, -0.25, 0, 0.25, 0.5])  # 2 blocks each side
    sg = ax_res.images[0]  # spectrogram under the carrier: 5 columns, Spectrogram cells kept
    assert sg.get_array().shape[1] == 5 and ax_res.get_ylim() == (-20, 20) and sg.get_clim() == (10, 90)
    tb_fmin.set_val("")  # one side blank: that side +- Nyquist
    assert ax.get_xlim() == (-1000, 50) and ax_res.get_ylim() == (-20, 20)
    tb_fmin.set_val("50")  # min >= max: message, nothing drawn
    assert ax.get_title().startswith("Epoch: ISO") and "min < max" in ax.get_title() and not ax.lines
    assert not cax.get_visible()  # message: colorbar hidden
    cells =(tb_fmin, tb_fmax, tb_pmin, tb_pmax, tb_sfmin, tb_sfmax, tb_spmin, tb_spmax)
    for tb in cells:
        tb.set_val("")
    for tb in (tb_pmin, tb_spmin):
        for bad in ("abc", "inf", "nan"):  # other cells blank: message, not an exception from set_ylim
            tb.set_val(bad)
            assert "min < max" in ax.get_title() and not ax.lines
        tb.set_val("")
    for tb, v in ((tb_fmin, "1000"), (tb_fmax, "-1000"), (tb_pmin, "1e6"), (tb_pmax, "-1e6"),
                  (tb_sfmin, "1000"), (tb_sfmax, "-1000"), (tb_spmin, "1e6"), (tb_spmax, "-1e6")):
        tb.set_val(v)  # alone, beyond the automatic other side: message, not an inverted axis
        assert "min < max" in ax.get_title() and not ax.lines and not ax_res.images, tb.label.get_text()
        tb.set_val("")
    sg = np.asarray(ax_res.images[0].get_array())
    assert ax.get_xlim() == (-1000, 1000) and ax_res.images[0].get_clim() == (np.nanmedian(sg), np.nanmax(sg))
    assert cax.get_visible() and len(fig.axes) == n_axes
    np.testing.assert_allclose(cax.get_ylim(), ax_res.images[0].get_clim(), rtol=1e-12)  # 1 ulp off on macOS arm64

    overlay = ax_res.lines[:2]  # carrier dots and rate dashes over the spectrogram
    assert cb.get_status() == [True] and all(ln.get_visible() for ln in overlay)
    cb.set_active(0)  # untick: hidden
    assert not any(ln.get_visible() for ln in overlay) and all(ln.get_visible() for ln in ax_res.lines[2:])
    b_next._observers.process("clicked", None)  # one T forward
    assert tb_epoch.text.endswith(":01.25")
    assert not any(ln.get_visible() for ln in ax_res.lines[:2])  # still hidden after a redraw
    cb.set_active(0)
    assert all(ln.get_visible() for ln in ax_res.lines[:2])

    from matplotlib.backend_bases import MouseEvent

    readout = fig.texts[-1]  # values under the mouse
    for a, want in ((ax, "f = -10.000 Hz   P = "), (ax_res, " dB")):
        x, y = a.transData.transform((-10, a.get_ylim()[0] + 1) if a is ax else (0.1, 0))  # not at an edge pixel
        MouseEvent("motion_notify_event", fig.canvas, x, y)._process()
        assert want in readout.get_text()

    tb_epoch.set_val(_fmt(t0 - np.timedelta64(10, "s")))
    assert ax.get_title().startswith("Window outside the file") and not ax.lines
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_browse_carrier_padding_and_cache(monkeypatch):
    rb = sys.modules["rdefpy.browse"]  # rdefpy.browse is the function

    calls = []
    real = rb.extract_carrier
    monkeypatch.setattr(rb, "extract_carrier", lambda *a, **k: calls.append(k["zero_padding"]) or real(*a, **k))
    pf = read_product(REF)
    fig = browse(pf, pf[0].header.timetag + np.timedelta64(1, "s"), integration_time=0.5, blocks=1)  # 3 blocks
    tb_pad, hann = fig._rdefpy_widgets[5], fig._rdefpy_widgets[17]
    assert calls == [1, 1, 1]  # carrier zero padding = the box
    hann._observers.process("clicked", None)
    assert len(calls) == 3  # window: display only, carrier cached
    tb_pad.set_val("2")
    assert calls[3:] == [2, 2, 2]  # pad change reruns the carrier
    monkeypatch.setattr(rb, "_CACHE_BYTES", 0)  # oldest freed on every insert
    tb_pad.set_val("3")
    tb_pad.set_val("2")  # pad-2 blocks were freed: computed again
    new = calls[6:]
    assert set(new[: new.index(2)]) == {3} and new.count(2) >= 3  # all 3 pad-2 blocks recomputed
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_browse_tab_moves_between_cells():
    from matplotlib.backend_bases import KeyEvent

    pf = read_product(REF)
    fig = browse(pf)
    w = fig._rdefpy_widgets
    order = [w[i] for i in (0, 1, 2, 5, 11, 12, 13, 14, 7, 8, 9, 10)]  # epoch, T, blocks, pad, Spectrogram, Periodogram
    press = lambda key: KeyEvent("key_press_event", fig.canvas, key)._process()
    typing = lambda: [tb for tb in order if tb.capturekeystrokes]
    epoch0 = order[0].text

    press("tab")  # no cell being typed in: nothing
    assert typing() == []
    tb_T = order[1]
    tb_T.begin_typing()
    tb_T.cursor_index = len(tb_T.text)
    for key in ("backspace", "0", ".", "5", "right"):  # right: inside the cell, not an epoch step
        press(key)
    assert tb_T.text == "0.5" and order[0].text == epoch0
    press("tab")  # applies T like Enter, then types in Blocks with the cursor at the end
    assert "T = 0.5 s" in fig.axes[0].get_title() and "\t" not in tb_T.text
    assert typing() == [order[2]] and order[2].cursor_index == len(order[2].text)
    press("shift+tab")
    assert typing() == [tb_T]
    for _ in range(len(order) - 1):  # forward from T wraps past the last cell to Epoch
        press("tab")
    assert typing() == [order[0]]
    press("shift+tab")  # and back from Epoch to the last cell
    assert typing() == [order[-1]]
    order[-1].stop_typing()
    plt.close(fig)


def test_chebwin_sidelobes():
    for m in (63, 64):
        w = _chebwin(m)
        assert w.max() == 1 and np.allclose(w, w[::-1])
        db = 20 * np.log10(np.abs(np.fft.fft(w, 64 * m)) / w.sum() + 1e-300)
        k = np.argmax(np.diff(db[: 64 * m // 2]) > 0)  # end of the main lobe
        assert db[k : 64 * m // 2].max() < -99  # 100 dB design


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")  # AGG, defined further down
def test_windows_at_a_gap_show_a_message(tmp_path):
    """4 records: s, s+1, (gap), s+3 flagged invalid, s+4. Windows at the gap's edges show the gap message."""
    import struct
    raw = (Path(__file__).parent / "reference" / "sample_16bit.prd").read_bytes()
    rec = raw[: struct.unpack_from("<I", raw, 4)[0]]
    s = struct.unpack_from("<I", rec, 44)[0]
    out = b""
    for k, flag in ((0, 0), (1, 0), (3, 1), (4, 0)):
        r = bytearray(rec)
        struct.pack_into("<I", r, 44, s + k)  # second of day
        struct.pack_into("<H", r, 20, flag)  # validity flag
        out += r
    (tmp_path / "gap.prd").write_bytes(out)
    pf = read_product(tmp_path / "gap.prd")
    t0 = pf[0].header.timetag
    fig = browse(pf)
    ax = fig.axes[0]
    for mid, title in ((1.75, "gap"), (3.0, "gap"), (4.0, "2000 samples")):  # record|gap, gap|record, invalid|valid
        fig._rdefpy_widgets[0].set_val(_fmt(t0 + np.timedelta64(round(mid * 1e9), "ns")))
        assert title in ax.get_title()
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_browse_header_lines_fit():
    """Agg hints monospace advances to whole pixels: the 0.64-em first guess (7.7 pt here, below _HDR_PT)
    draws 7 px per char, not 6.4, so the size steps down until the longest line fits the panel."""
    fig = browse(read_product(REF))
    img = np.asarray(fig.images[0].get_array())  # the header pixels, 1:1 in the panel
    ink = np.flatnonzero(img[..., :3].max(axis=(0, 2)) > 40)
    assert ink[-1] < img.shape[1] - 2  # blank margin right of the widest line
    plt.close(fig)


# =============================================================================
# Synthetic tone files: what is drawn, against values worked out here
# =============================================================================

FS, F_TONE = 1000, 123.0  # 1 kHz records; tone on a 1 Hz bin for T = 1 s, period the whole record
PEAK_DB = 20 * np.log10(24000)  # codes k = round(12000 cos): samples 2k+1 (§5.3.6), amplitude 24000
NOON = np.datetime64("2026-02-08T12:00:00", "ns")  # rdef_spec defaults: 2026, day 39, 43200 s
AGG = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


def _tone_record(s, fs=FS, **hdr) -> bytes:
    """One 16-bit record starting *s* seconds after noon: a tone at F_TONE, phase continuous in time."""
    ph = 2 * np.pi * F_TONE * (s + np.arange(fs) / fs)
    k_i, k_q = (np.round(12000 * fn(ph)).astype(int).tolist() for fn in (np.cos, np.sin))
    return spec.record_bytes(k_i, k_q, sample_size=16, sample_rate=fs, time_tag_second_of_day=43200 + s, **hdr)


def _click(button):
    button._observers.process("clicked", None)


def _no_render(fig):
    """Skip the Agg render after each change (most of a redraw's time): for tests that read the artists only."""
    fig.canvas.draw_idle = lambda *args, **kwargs: None
    return fig


def _press(fig, key):
    from matplotlib.backend_bases import KeyEvent

    KeyEvent("key_press_event", fig.canvas, key)._process()


@AGG
def test_spectrum_is_the_periodogram_of_the_samples_as_recorded(write_prd):
    """abs(FFT(x w))^2 / sum(w)^2 of the recorded samples, for every window of the
    menu: nothing removed (the DC bin keeps the +1 +1j of the 2k+1 codes), and an
    on-bin tone of amplitude A peaks at 20 log10 A whatever the window."""
    from rdefpy.browse import _WINDOWS

    pf = read_product(write_prd(*(_tone_record(s) for s in range(3))))
    fig = _no_render(browse(pf, blocks=0))
    ax, w = fig.axes[0], fig._rdefpy_widgets
    tb_pad, b_win, opts = w[5], w[6], w[16:]
    x = pf.samples[:FS]  # the window: all of record 0
    assert abs(x.mean() - (1 + 1j)) < 0.1  # the tone averages out; the code offset does not

    def want(win, pad=1):
        p = np.fft.fftshift(np.abs(np.fft.fft(x * win, pad * FS)) ** 2) / win.sum() ** 2
        return 10 * np.log10(np.maximum(p, np.finfo(float).tiny))  # zero power is drawn at the float floor

    def drawn():
        f, db = ax.lines[0].get_xydata().T
        return f, db

    dot = ax.lines[1].get_xydata().copy()  # the carrier estimate: unwindowed, unpadded samples
    np.testing.assert_allclose(dot, [[F_TONE, PEAK_DB]], rtol=0, atol=1e-3)
    assert [b.label.get_text() for b in opts] == list(_WINDOWS) == [
        "none", "Hann", "Hamming", "Blackman", "Kaiser 8", "Chebyshev 100 dB"]
    for b in opts:
        name = b.label.get_text()
        _click(b_win)  # open the menu, pick the option
        _click(b)
        assert b_win.label.get_text() == f"Window: {name} ▾"
        f, db = drawn()
        np.testing.assert_array_equal(f, np.arange(-FS // 2, FS // 2))  # 1 Hz bins, channel centre at 0
        np.testing.assert_allclose(db, want(_WINDOWS[name](FS)), rtol=0, atol=1e-9, err_msg=name)
        assert f[db.argmax()] == F_TONE and db.max() == pytest.approx(PEAK_DB, abs=0.01), name
        np.testing.assert_array_equal(ax.lines[1].get_xydata(), dot)  # the window never reaches the carrier
        if name == "none":
            assert db[f == 0][0] == pytest.approx(10 * np.log10(2), abs=0.1)  # DC: |1 + 1j|^2, not removed

    tb_pad.set_val("4")  # zero padding: a finer grid of the same samples (4000 bins: none pooled)
    f, db = drawn()
    np.testing.assert_array_equal(f, np.arange(-2 * FS, 2 * FS) / 4)
    np.testing.assert_allclose(db, want(_WINDOWS[name](FS), 4), rtol=0, atol=1e-9)
    assert "max of every" not in ax.get_title()
    plt.close(fig)


@AGG
def test_arrows_step_the_epoch_and_unreadable_cells_show_the_message(write_prd):
    from rdefpy.browse import _BAD_INPUT

    pf = read_product(write_prd(*(_tone_record(s) for s in range(3))))
    fig = _no_render(browse(pf, "2026-02-08T12:00:01.5", blocks=0))  # epoch as text: middle of record 1
    ax = fig.axes[0]
    tb_epoch, tb_T, tb_n, b_prev, b_next, tb_pad = fig._rdefpy_widgets[:6]
    at = lambda: tb_epoch.text.removeprefix("2026-02-08T12:00:")  # noqa: E731
    assert at() == "01.5" and "T = 1 s   (1000 samples)" in ax.get_title()

    for act, where in ((lambda: _press(fig, "right"), "02.5"), (lambda: _press(fig, "left"), "01.5"),
                       (lambda: _press(fig, "up"), "01.5"), (lambda: _click(b_prev), "00.5"),
                       (lambda: _click(b_next), "01.5"), (lambda: _press(fig, "left"), "00.5")):
        act()
        assert at() == where and ax.get_title().startswith(f"2026-02-08T12:00:{where} ") and len(ax.lines) == 2
    _press(fig, "left")  # one more: the box moves, the window is before the file
    assert tb_epoch.text == "2026-02-08T11:59:59.5" and not ax.lines
    assert ax.get_title() == "Window outside the file (2026-02-08T12:00:00 to 2026-02-08T12:00:03)"
    _press(fig, "right")
    assert at() == "00.5" and len(ax.lines) == 2

    tb_T.set_val("0.5")  # the step is T
    _press(fig, "right")
    assert at() == "01" and "T = 0.5 s   (500 samples)" in ax.get_title()
    tb_epoch.set_val("2026-02-08T12:00:01.25Z")  # a trailing Z is accepted
    assert ax.get_title().startswith("2026-02-08T12:00:01.25 ")

    for tb, bad, good in ((tb_T, "0", "1"), (tb_T, "one", "1"), (tb_n, "-1", "0"), (tb_pad, "0", "1"),
                          (tb_pad, "2.5", "1"), (tb_epoch, "noon", "2026-02-08T12:00:01.5")):
        tb.set_val(bad)
        assert ax.get_title() == _BAD_INPUT and not ax.lines, (tb.label.get_text(), bad)
        epoch = tb_epoch.text
        _press(fig, "right")  # nothing to step from
        assert tb_epoch.text == epoch
        tb.set_val(good)
        assert len(ax.lines) == 2, (tb.label.get_text(), good)

    tb_T.set_val("0.0005")  # half a sample
    assert ax.get_title() == "T * sample_rate must be whole (1000 Hz)" and not ax.lines
    plt.close(fig)


@AGG
def test_values_no_window_can_have_show_the_message():
    from rdefpy.browse import _BAD_INPUT

    fig = _no_render(browse(read_product(REF), blocks=0))
    ax, wrong = fig.axes[0], []
    for box, text in ((1, "inf"), (1, "1e30"), (2, "99999999999999999999"), (0, ""), (0, "NaT")):
        tb = fig._rdefpy_widgets[box]
        good = tb.text
        try:
            tb.set_val(text)
            if ax.get_title() != _BAD_INPUT or ax.lines:
                wrong.append((tb.label.get_text(), text, ax.get_title()))
        except OverflowError as e:
            wrong.append((tb.label.get_text(), text, repr(e)))
        tb.set_val(good)
        assert len(ax.lines) == 2  # readable again: drawn
    plt.close(fig)
    assert not wrong


@AGG
def test_blocks_beyond_the_file_draw_as_far_as_it_goes():
    """A block count far past the file (its ns span overflows int64) draws the blocks the file has."""
    fig = _no_render(browse(read_product(REF), blocks=0))
    ax_res = fig.axes[2]
    fig._rdefpy_widgets[2].set_val("10000000000")
    assert fig.axes[0].get_title().startswith("2026-02-08T12:00:00.5 ") and len(ax_res.lines[0].get_xdata()) == 2
    plt.close(fig)


@AGG
def test_zero_pad_is_1_to_10():
    from rdefpy.browse import _BAD_INPUT

    fig = _no_render(browse(read_product(REF), blocks=0))
    ax, tb_pad = fig.axes[0], fig._rdefpy_widgets[5]
    tb_pad.set_val("10")
    assert len(ax.lines) == 2 and "2026-02-08T12:00:00.5 " in ax.get_title()
    tb_pad.set_val("11")
    assert ax.get_title() == _BAD_INPUT and not ax.lines
    plt.close(fig)


@AGG
def test_blocks_over_a_gap_or_an_invalid_record_are_blank(write_prd):
    """Seconds 0, 1, (2 missing), 3 flagged invalid, 4, 5; six 1 s blocks. No
    estimate and no spectrum across the gap; no estimate from the invalid
    record, whose spectrum is still shown. Nothing is interpolated."""
    pf = read_product(write_prd(*(_tone_record(s, validity_flag=int(s == 3)) for s in (0, 1, 3, 4, 5))))
    fig = _no_render(browse(pf, blocks=5))  # the window is record 0; the file ends 5 blocks later
    ax, ax_sky, ax_res = fig.axes[:3]
    t = ax_sky.lines[0].get_xdata()
    np.testing.assert_array_equal(t, np.arange(6.0))  # block midpoints, s from the epoch
    for line, value in ((ax_sky.lines[0], F_TONE), (ax_sky.lines[1], 0.0), (ax_res.lines[0], F_TONE)):
        y = line.get_ydata()  # before downconversion and f_DC (both minus the fixed part), baseband
        assert np.isnan(y).tolist() == [False, False, True, True, False, False]
        np.testing.assert_allclose(y[[0, 1, 4, 5]], value, rtol=0, atol=1e-3)
    sg = np.asarray(ax_res.images[0].get_array())
    assert sg.shape == (FS, 6)
    assert np.isnan(sg).all(axis=0).tolist() == [False, False, True, False, False, False]
    assert not np.isnan(sg[:, [0, 1, 3, 4, 5]]).any()
    np.testing.assert_allclose(sg[:, [0, 1, 3, 4, 5]].max(axis=0), PEAK_DB, rtol=0, atol=0.01)
    assert ax_res.images[0].get_clim()[1] == pytest.approx(PEAK_DB, abs=0.01)  # automatic: ... to the maximum
    plt.close(fig)


@AGG
@pytest.mark.filterwarnings("ignore:Metadata fields vary:RuntimeWarning")
def test_window_over_a_sample_rate_change_shows_a_message(write_prd):
    pf = read_product(write_prd(_tone_record(0), _tone_record(1, fs=2 * FS)))
    fig = _no_render(browse(pf, NOON + np.timedelta64(1, "s"), blocks=0))  # half in each record
    ax = fig.axes[0]
    assert ax.get_title() == "Window crosses a sample-rate change" and not ax.lines
    fig._rdefpy_widgets[0].set_val(_fmt(NOON + np.timedelta64(1500, "ms")))  # inside the 2 kHz record
    assert "(2000 samples)" in ax.get_title() and ax.get_xlim() == (-1000, 1000)
    plt.close(fig)


@AGG
def test_readout_under_the_mouse_and_header_scrolling(write_prd):
    from matplotlib.backend_bases import MouseEvent, ResizeEvent

    pf = read_product(write_prd(*(_tone_record(s) for s in range(3))))
    fig = browse(pf, blocks=1)
    ax, ax_sky, ax_res, ax_txt = fig.axes[:4]
    readout = next(t for t in fig.texts if t.get_animated())

    def mouse(name, a, xy, **kw):
        MouseEvent(name, fig.canvas, *a.transData.transform(xy), **kw)._process()

    for cell, value in zip(fig._rdefpy_widgets[11:13], ("120", "127")):  # Spectrogram f min, f max: 7 bins,
        cell.set_val(value)  # each many pixels tall (the readout takes the cell at the mouse's whole pixel)
    fig.canvas.draw()
    mouse("motion_notify_event", ax_sky, (0.25, 60.0))
    assert readout.get_text() == "t = +0.250 s from epoch   f = 60.0000 Hz + f_DC,fixed"
    mouse("motion_notify_event", ax_res, (0.25, F_TONE + 0.5))  # inside the tone's bin of the spectrogram
    assert readout.get_text() == f"t = +0.250 s from epoch   f = {F_TONE + 0.5:.3f} Hz   P = {PEAK_DB:.1f} dB"
    MouseEvent("motion_notify_event", fig.canvas, 1, 1)._process()  # outside every panel
    assert readout.get_text() == ""

    # Headers: taller than the panel once the figure is short; the wheel moves them by whole lines.
    text = ax_txt.texts[0].get_text()  # one record under the window: its whole header, nothing "changed"
    assert text.startswith("Record 2026-02-08T12:00:00\nRecordHeader\n") and "changed" not in text
    assert "record_length: 4,176 bytes" in text and len(text.splitlines()) == 24  # title, class, 21 fields, timetag
    wheel = lambda step: MouseEvent("scroll_event", fig.canvas, *ax_txt.transAxes.transform((0.5, 0.5)),  # noqa: E731
                                    step=step)._process()
    shown = lambda: np.array(fig.images[0].get_array())  # noqa: E731
    whole = shown()
    wheel(-1)
    np.testing.assert_array_equal(shown(), whole)  # everything fits: nothing to scroll
    fig.set_size_inches(17, 2.5)
    ResizeEvent("resize_event", fig.canvas)._process()
    fig.canvas.draw()
    top = shown()
    assert len(top) == int(ax_txt.bbox.height) < len(whole)
    np.testing.assert_array_equal(top, whole[: len(top)])  # the same pixels, cut to the panel
    wheel(-1)  # down
    down = shown()
    rows = next(d for d in range(1, len(whole) - len(top) + 1) if np.array_equal(down, whole[d : d + len(top)]))
    assert rows > 8  # more than one text line
    wheel(1)
    np.testing.assert_array_equal(shown(), top)
    wheel(1)  # already at the top
    np.testing.assert_array_equal(shown(), top)
    for _ in range(40):
        wheel(-1)
    np.testing.assert_array_equal(shown(), whole[-len(top) :])  # stops at the last line
    plt.close(fig)


@AGG
def test_spectrum_beyond_4096_bins_is_max_pooled(write_prd):
    """Zero pad 10: 10,000 bins drawn as the max of every 3 (display only). The peak is
    drawn at the tone, at its full height; the last 10,000 % 3 = 1 bin (+499.9 Hz),
    which makes no whole group, is not drawn."""
    pf = read_product(write_prd(*(_tone_record(s) for s in range(2))))
    fig = _no_render(browse(pf, blocks=0))
    ax = fig.axes[0]
    fig._rdefpy_widgets[5].set_val("10")
    assert "max of every 3 bins" in ax.get_title()
    f, db = ax.lines[0].get_xydata().T
    x = pf.samples[:FS]
    p = np.fft.fftshift(np.abs(np.fft.fft(x, 10 * FS)) ** 2) / FS**2
    full = np.fft.fftshift(np.fft.fftfreq(10 * FS, 1 / FS))
    i = p[:9999].reshape(-1, 3).argmax(1) + np.arange(0, 9999, 3)  # each group's peak bin
    assert len(f) == 3333 and f[-1] == full[9996:9999][p[9996:9999].argmax()] and full[-1] == pytest.approx(499.9)
    np.testing.assert_array_equal(f, full[i])
    np.testing.assert_allclose(db, 10 * np.log10(p[i]), rtol=0, atol=1e-9)
    assert f[db.argmax()] == F_TONE and db.max() == pytest.approx(PEAK_DB, abs=0.01)
    plt.close(fig)


@AGG
def test_tab_skips_the_epoch_box_under_the_open_menu_and_reads_shift(write_prd):
    """Shift+Tab arrives as "shift+tab", "backtab", "shift+iso_left_tab", or (Tk) "tab"
    with bit 0 of the GUI event's state set; any other state goes forward. The
    open window menu covers the Epoch box: Tab passes over it."""
    from types import SimpleNamespace

    from matplotlib.backend_bases import KeyEvent

    fig = _no_render(browse(read_product(write_prd(*(_tone_record(s) for s in range(2)))), blocks=0))
    w = fig._rdefpy_widgets
    tb_epoch, tb_T, tb_n, tb_pad, last = w[0], w[1], w[2], w[5], w[10]  # last of the Tab order: Periodogram P max
    typing = lambda: [tb for tb in w if getattr(tb, "capturekeystrokes", False)]  # noqa: E731

    def press(key, state=None):
        gui = None if state is None else SimpleNamespace(state=state)
        KeyEvent("key_press_event", fig.canvas, key, guiEvent=gui)._process()

    for key, state, want in (("backtab", None, tb_T), ("shift+iso_left_tab", None, tb_T), ("tab", 1, tb_T),
                             ("tab", 0x11, tb_T), ("tab", 0, tb_pad), ("tab", "Shift", tb_pad),
                             ("tab", None, tb_pad)):  # from Blocks: back to T, forward to Zero pad
        tb_n.begin_typing()
        press(key, state)
        assert typing() == [want], (key, state)
        want.stop_typing()
    _click(w[6])  # open the window menu: Epoch inactive
    assert not tb_epoch.active
    tb_T.begin_typing()
    press("shift+tab")
    assert typing() == [last]  # T back to the last cell, over Epoch
    press("tab")
    assert typing() == [tb_T]  # and forward from the last cell to T
    tb_T.stop_typing()
    plt.close(fig)


@AGG
def test_scrollbar_press_and_drag_move_the_headers(write_prd):
    from matplotlib.backend_bases import MouseEvent, ResizeEvent

    fig = browse(read_product(write_prd(*(_tone_record(s) for s in range(2)))), blocks=0)
    ax_txt = fig.axes[3]
    shown = lambda: np.array(fig.images[0].get_array())  # noqa: E731
    whole = shown()
    fig.set_size_inches(17, 2.5)  # headers taller than the panel: a scrollbar
    ResizeEvent("resize_event", fig.canvas)._process()
    fig.canvas.draw()
    b = ax_txt.bbox
    top, h = shown(), int(b.height)
    assert len(top) == h < len(whole)
    mouse = lambda name, x, y: MouseEvent(name, fig.canvas, x, y, button=1)._process()  # noqa: E731
    mouse("button_press_event", b.x1 - 3, b.y0 + 1)  # bottom of the bar: the last lines
    np.testing.assert_array_equal(shown(), whole[-h:])
    mouse("motion_notify_event", b.x1 - 3, b.y1 - 1)  # dragged to the top: the first lines
    np.testing.assert_array_equal(shown(), top)
    mouse("button_release_event", b.x1 - 3, b.y1 - 1)
    mouse("motion_notify_event", b.x1 - 3, b.y0 + 1)  # released: moving the mouse moves nothing
    np.testing.assert_array_equal(shown(), top)
    mouse("button_press_event", b.x0 + 20, b.y0 + 1)  # in the text, not on the bar
    mouse("motion_notify_event", b.x0 + 20, b.y0 + 1)
    np.testing.assert_array_equal(shown(), top)
    plt.close(fig)


@AGG
@pytest.mark.parametrize("length", [10, 80, 120, 1000])
def test_header_font_fits_the_longest_line(length, write_prd, monkeypatch):
    """8 pt while the longest line fits; smaller, by 0.1 pt, until it does, stopping
    at the first size at or below 4 pt even if it still does not fit. Not a strict
    4 pt floor: a line too long for 4 pt starts below 4 (w * 72 / (dpi * 0.64 * len))."""
    from rdefpy.browse import _BAR_PX, _text_width
    from rdefpy.prd import _display

    rb = sys.modules["rdefpy.browse"]
    pts, real = [], rb._text_image
    monkeypatch.setattr(rb, "_text_image", lambda text, w, dpi, pt: pts.append((w, dpi, pt)) or real(text, w, dpi, pt))
    monkeypatch.setattr(_display, "header_str", lambda h: "RecordHeader\n" + "x" * length)
    fig = _no_render(browse(read_product(write_prd(_tone_record(0))), blocks=0))
    w, dpi, pt = pts[-1]
    assert w == fig.axes[3].bbox.width - _BAR_PX - 4
    if length == 10:
        assert pt == 8.0
    elif length == 80:  # the 0.64 em estimate already fits
        assert pt == pytest.approx(w * 72 / (dpi * 0.64 * length)) and 4 < pt < 8
        assert _text_width("x" * length, dpi, pt) <= w
    elif length == 120:
        assert 3.9 < pt <= 4 and _text_width("x" * length, dpi, pt + 0.1) > w
    else:
        assert pt == pytest.approx(w * 72 / (dpi * 0.64 * length)) and pt < 4
    plt.close(fig)


def test_browse_without_matplotlib_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)  # import fails
    with pytest.raises(ImportError, match=r'pip install "rdefpy\[browse\]"'):
        browse(read_product(REF))


def test_empty_product_file_is_refused():
    from rdefpy import ProductFile

    before = plt.get_fignums()
    with pytest.raises(ValueError, match="no records"):
        browse(ProductFile())
    assert plt.get_fignums() == before  # no window left open


@pytest.mark.parametrize("lo, hi, want", [("", "", (None, None)), ("-5", "", (-5.0, None)), ("", " 2.5 ", (None, 2.5)),
                                          ("-1e3", "1e3", (-1000.0, 1000.0)), (" ", "0", (None, 0.0))])
def test_limit_cells_blank_is_automatic(lo, hi, want):
    from rdefpy.browse import _limits

    assert _limits(lo, hi) == want


@pytest.mark.parametrize("lo, hi", [("2", "1"), ("1", "1"), ("inf", ""), ("", "-inf"), ("nan", ""), ("1", "nan"),
                                    ("abc", ""), ("", "1,5")])
def test_limit_cells_refuse_what_is_not_a_range(lo, hi):
    from rdefpy.browse import _limits

    with pytest.raises(ValueError):
        _limits(lo, hi)