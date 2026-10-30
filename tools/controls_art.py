#!/usr/bin/env python3
"""The on-screen controls' art: every control as a PNG, normal and held, into app/src/main/res/drawable-nodpi, drawn
by TouchControls.java. Each is drawn here as an SVG and rendered by rsvg-convert, the line icons for what a button
does in the game (native/ui/<name>.svg, 24 x 24, white strokes) set into it.

Every image is a control of radius 100 in a 260 x 260 box centred on it, so TouchControls draws each one 1.3 times its
radius out from its centre: the room above a button takes its badge. Run this after changing the look or an icon and
commit the PNGs. Needs rsvg-convert and Noto Sans."""
import base64
import math
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "native" / "ui"
OUT = ROOT / "app" / "src" / "main" / "res" / "drawable-nodpi"
PX = 256          # the PNG's side
C, R = 130, 100   # the control's centre and radius in the box

# the look: a thin light outline over next to no backing, so the battlefield shows through; held, the inside washes white
BACK, BACK_ON = "fill='#000' fill-opacity='0.12'", "fill='#fff' fill-opacity='0.24'"
RIM, RIM_ON = ("#ebebeb", 0.65, 2.6), ("#ffffff", 0.98, 3.8)
INK, INK_ON = 0.88, 1.0
FONT = "font-family='Noto Sans' font-weight='800' fill='#fff' text-anchor='middle' dominant-baseline='central'"


def svg(body):
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 260 260' width='{PX}' height='{PX}'>{body}</svg>"


def rim(on, r=R, gap=0.0):
    colour, opacity, width = RIM_ON if on else RIM
    rr = r - width / 2
    style = f"fill='none' stroke='{colour}' stroke-opacity='{opacity}' stroke-width='{width}' stroke-linecap='round'"
    if gap <= 0:
        return f"<circle cx='{C}' cy='{C}' r='{rr}' {style}/>"
    half = math.asin(min(gap / rr, 1.0))
    a0, a1 = -math.pi / 2 + half, 3 * math.pi / 2 - half
    x0, y0 = C + rr * math.cos(a0), C + rr * math.sin(a0)
    x1, y1 = C + rr * math.cos(a1), C + rr * math.sin(a1)
    return f"<path d='M{x0:.2f} {y0:.2f}A{rr} {rr} 0 1 1 {x1:.2f} {y1:.2f}' {style}/>"


def backing(on, badge_r=0.0):
    """The disc, and a badge's on its top edge with it as one shape, filled once: two see-through fills laid over each
    other show their overlap."""
    fill = BACK_ON if on else BACK
    if badge_r <= 0:
        return f"<circle cx='{C}' cy='{C}' r='{R}' {fill}/>"
    cut = 2 * math.asin(badge_r / (2 * R))
    a0, a1 = -math.pi / 2 + cut, 3 * math.pi / 2 - cut
    rx, ry = C + R * math.cos(a0), C + R * math.sin(a0)
    lx, ly = C + R * math.cos(a1), C + R * math.sin(a1)
    return (f"<path d='M{rx:.2f} {ry:.2f}A{R} {R} 0 1 1 {lx:.2f} {ly:.2f}"
            f"A{badge_r} {badge_r} 0 1 1 {rx:.2f} {ry:.2f}Z' {fill}/>")


def badge(on, br, text, size):
    colour, opacity, width = RIM_ON if on else RIM
    by = C - R
    return (f"<circle cx='{C}' cy='{by}' r='{br - width / 2:.2f}' fill='none' stroke='{colour}' "
            f"stroke-opacity='{opacity}' stroke-width='{width}'/>"
            f"<text x='{C}' y='{by + 1}' font-size='{size}' fill-opacity='{INK_ON if on else INK}' {FONT}>{text}</text>")


def icon(name, size, dy, on):
    data = base64.b64encode((ICONS / f"{name}.svg").read_bytes()).decode()
    x, y = C - size / 2, C - size / 2 + dy
    return (f"<image x='{x:.2f}' y='{y:.2f}' width='{size}' height='{size}' opacity='{INK_ON if on else INK}' "
            f"href='data:image/svg+xml;base64,{data}'/>")


def face(letter, name, on):
    br = R * 0.26
    return svg(backing(on, br) + rim(on, gap=br * 1.35) + icon(name, R * 1.3, R * 0.04, on) + badge(on, br, letter, 34))


def shoulder(label, name, on):
    br = R * 0.30
    return svg(backing(on, br) + rim(on, gap=br * 1.35) + icon(name, R * 1.15, R * 0.06, on) + badge(on, br, label, 28))


def dotted(white, name, on):
    """MAP and ORDERS, the pad's Black and White: what each does, and the pad's button as a dot on the badge."""
    br = R * 0.26
    fill, edge = ("#f0eee8", "#000") if white else ("#040404", "#fff")
    return svg(backing(on, br) + rim(on, gap=br * 1.35) + icon(name, R * 1.15, R * 0.06, on) + badge(on, br, "", 1) +
               f"<circle cx='{C}' cy='{C - R}' r='{br * 0.5:.2f}' fill='{fill}' stroke='{edge}' stroke-opacity='0.5' "
               f"stroke-width='3'/>")


def label(text, size, on):
    return svg(backing(on) + rim(on) +
               f"<text x='{C}' y='{C + 2}' font-size='{size}' letter-spacing='1' fill-opacity='{INK_ON if on else INK}' "
               f"{FONT}>{text}</text>")


def chevron(cx, cy, dx, dy, t, width, opacity):
    tip = (cx + dx * t, cy + dy * t)
    b1 = (cx - dx * t * 0.4 - dy * t, cy - dy * t * 0.4 + dx * t)
    b2 = (cx - dx * t * 0.4 + dy * t, cy - dy * t * 0.4 - dx * t)
    return (f"<polyline points='{b1[0]:.2f},{b1[1]:.2f} {tip[0]:.2f},{tip[1]:.2f} {b2[0]:.2f},{b2[1]:.2f}' "
            f"fill='none' stroke='#fff' stroke-opacity='{opacity}' stroke-width='{width}' stroke-linecap='round' "
            f"stroke-linejoin='round'/>")


def arrow(dx, dy, on):
    return svg(backing(on) + rim(on) + chevron(C, C, dx, dy, R * 0.38, 7, INK_ON if on else INK))


def stick_base():
    out = backing(False) + rim(False)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        out += chevron(C + dx * R * 0.8, C + dy * R * 0.8, dx, dy, R * 0.09, 2.5, RIM[1])
    return svg(out)


def gear(on):
    # eight teeth, a flat top at the outer radius and a flat valley at the inner one, a hub in the middle
    pts = []
    step = 2 * math.pi / 8
    outer, inner = R * 0.66, R * 0.49
    for i in range(8):
        for k, rr in ((-0.30, inner), (-0.17, outer), (0.17, outer), (0.30, inner)):
            a = i * step + k * step
            pts.append(f"{C + math.cos(a) * rr:.2f},{C + math.sin(a) * rr:.2f}")
    style = f"fill='none' stroke='#fff' stroke-opacity='{INK_ON if on else INK}' stroke-width='8.5' stroke-linejoin='round'"
    return svg(backing(on) + rim(on) + f"<polygon points='{' '.join(pts)}' {style}/>"
               f"<circle cx='{C}' cy='{C}' r='{R * 0.21}' {style}/>")


def art():
    out = {}
    for on in (False, True):
        s = "_on" if on else ""
        # A jump, B musou, X attack, Y charge: the PS2 game's Cross, Circle, Square, Triangle
        for letter, name in (("A", "jump"), ("B", "musou"), ("X", "attack"), ("Y", "charge")):
            out[f"ctl_{letter.lower()}{s}"] = face(letter, name, on)
        out[f"ctl_lt{s}"] = shoulder("LT", "guard", on)
        out[f"ctl_rt{s}"] = shoulder("RT", "bow", on)
        out[f"ctl_r3{s}"] = shoulder("R3", "rage", on)
        out[f"ctl_start{s}"] = label("START", 34, on)
        out[f"ctl_back{s}"] = label("BACK", 40, on)
        out[f"ctl_l3{s}"] = label("L3", 50, on)
        out[f"ctl_orders{s}"] = dotted(True, "orders", on)
        out[f"ctl_map{s}"] = dotted(False, "map", on)
        out[f"ctl_gear{s}"] = gear(on)
        for n, d in (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))):
            out[f"ctl_arrow_{n}{s}"] = arrow(*d, on)
        # the knob, a plain disc: the stick's and the right stick's, held while a finger pushes it
        out[f"ctl_knob{s}"] = svg(backing(on) + rim(on))
    out["ctl_stick"] = stick_base()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in art().items():
            src = Path(tmp) / f"{name}.svg"
            src.write_text(text)
            subprocess.run(["rsvg-convert", str(src), "-w", str(PX), "-h", str(PX), "-o", str(OUT / f"{name}.png")],
                           check=True)
    print(f"{OUT.relative_to(ROOT)}: {len(art())} images")


if __name__ == "__main__":
    main()
