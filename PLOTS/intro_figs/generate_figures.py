#!/usr/bin/env python3
"""Edit this file, then run it to generate both square figures.

    source /home/mmoslem3/ENV/bin/activate
    python /scratch/mmoslem3/MKDT/PLOTS/intro_figs/generate_figures.py

Everything about the plot is here: search for EDIT 1 through EDIT 7 below.
Only this file and the referenced images are inputs. TeX files are generated
automatically; the next run overwrites them. The existing TikZ renderer preserves
the exact appearance. Requires Python's standard library, pdflatex (TikZ,
preview, mathptmx), and Poppler's pdftoppm/pdfinfo/pdftotext. No GPU or inference.

Outputs: intro_left.pdf, intro_right.pdf, PNG previews, generated TeX/logs,
and intro.tex (the optional two-panel manuscript snippet).
"""
import math
import os
from pathlib import Path
import re
import shutil
from string import Template
import subprocess


# ==================== EDIT 1: FILES AND OUTPUT ====================
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE                       # Change to save figures elsewhere.
PREVIEW_PIXELS = 1200                   # PNG side length; PDF stays vector.
CHECK_LAYOUT = True                    # False permits intentional overlaps.

# ==================== EDIT 2: VALUES, IMAGES, AND ORDER ====================
# Positions run CLOCKWISE from the top. Each value matches the image in the
# same list position. Reorder the image names AND values to move whole pairs.
# Positions: top, upper-right, right-upper, right-lower, lower-right,
# bottom, lower-left, left-lower, left-upper, upper-left.
PROTOTYPE_FILES = [
    'prototype-01.png', 'prototype-02.png', 'prototype-03.png',
    'prototype-04.png', 'prototype-05.png', 'prototype-06.png',
    'prototype-07.png', 'prototype-08.png', 'prototype-09.png',
    'prototype-10.png',
]
# Displayed percentages, not recomputed measurements. Manual left-panel edits:
# 6.1->14.2, 33.5->29.5, 11.5->9.4, 8.6->6.6. The original measurements remain
# in mapping_tau0025_flip/assignments.json; editing this file never changes them.
PANELS = [
    {
        'name': 'intro_left',
        'image_dir': HERE / 'images/view1',
        'center_image': 'real.png',
        'images': PROTOTYPE_FILES.copy(),
        'values': [7.8, 4.5, 14.2, 2.7, 10.5, 29.5, 10.2, 9.4, 4.6, 6.6],
        'fill_rgb': (217, 230, 244),     # Light blue; RGB channels 0..255.
        'accent_rgb': (91, 132, 180),    # Bar outlines / highlighted frames.
        'style': {},                    # Override any STYLE key for this panel.
    },
    {
        'name': 'intro_right',
        'image_dir': HERE / 'images/view2',
        'center_image': 'real.png',
        'images': PROTOTYPE_FILES.copy(),
        'values': [5.7, 5.8, 19.9, 5.1, 10.7, 22.5, 10.2, 7.8, 3.3, 8.9],
        'fill_rgb': (248, 222, 218),     # Light red.
        'accent_rgb': (199, 106, 100),
        'style': {},
    },
]
# Optional per-panel 'angles': [90, 54, 18, ...] sets individual positions in
# degrees (0=right, 90=top). Otherwise images/bars are evenly spaced clockwise.
# Example: PANELS[1]['style'] = {'label_size': 10, 'image_radius': 108}

# ==================== EDIT 3: SIZE, SPACING, COLORS, AND FONTS ====================
# All lengths are PDF points (bp, 72 per inch); angles are degrees.
# Shared defaults: PANELS[i]['style'] can override any key for just one panel.
STYLE = {
    # Canvas and images. Center and rim images currently have the same size.
    'page_size': 252,                   # Square PDF side length / whitespace.
    'image_radius': 110,                # Center-to-center distance to rim images.
    'image_size': 28,
    'center_image_size': 28,
    'start_angle': 90,                  # First image/bar at the top.
    'frame_padding': .7,                # Gap from image edge to frame.
    'frame_width': .4,
    'highlight_threshold': 20,          # Values above this get accent frames.
    'highlight_frame_width': .65,
    'center_frame_width': .55,
    'center_frame_rgb': (91, 132, 180),
    'circle_radius': 24,                # Inner circle around the real image.
    'circle_width': .4,
    'circle_mix': 24,                   # % muted color mixed with white.
    'background_rgb': (255, 255, 255),
    'text_rgb': (40, 44, 49),
    'muted_rgb': (91, 101, 111),
    'frame_mix': 28,

    # Bars: tip radius = bar_inner + bar_length * value / scale_max.
    'bar_inner': 34,
    'bar_length': 52,                   # Length when value equals scale_max.
    'scale_max': 35,
    'bar_half_angle': 12,               # Total bar width = twice this angle.
    'bar_line_width': .4,
    'bar_outline_mix': 80,

    # Labels: bottom of each number faces its bar's outer edge.
    'label_size': 9.5,                  # SAME font size for every number.
    'label_line_height': 10,
    'label_gap': .6,                    # Gap between bar tip and label box.
    'label_padding': .5,
    'label_rotation_offset': -90,       # Rotation = bar angle + this offset.
    'label_anchor': 'south',            # south=bottom of label meets the tip.
    'number_format': '.1f',             # '.0f'=integers; '.2f'=two decimals.
    'label_suffix': '',                 # r'\%' adds percent symbols.
    'font_packages': 'mathptmx',        # Times. Alternative example: 'lmodern'.
    'font_family': r'\rmfamily',        # Serif; r'\sffamily' for sans serif.
    'label_weight': r'\bfseries',       # Bold; r'\mdseries' for normal weight.
    'label_shape': r'\upshape',         # Upright; r'\itshape' for italic.

    # Faint grid circles and spokes toward the surrounding images.
    'grid_radii': (60, 86),             # () removes these circles.
    'grid_mix': 17,
    'grid_width': .3,
    'grid_dash_on': .7,
    'grid_dash_off': 2,
    'spoke_start': 24,
    'spoke_end': 88,
    'spoke_width': .3,
}

# ==================== EDIT 4: OPTIONAL MANUSCRIPT CAPTION ====================
# Written to intro.tex; neither standalone panel contains a title or caption.
CAPTION = (
    r'\textbf{Soft assignments across augmentations.} '
    r'A frozen TinyImageNet ResNet-18 encoder assigns the real image to the ten '
    r'shown synthetic prototypes using softmax over $\ell_2$-normalized feature '
    r'similarities ($\tau=0.025$). Each view applies the same transform to all images. '
    r'Percentages are normalized over these ten prototypes; both panels use the '
    r'same 0--35\% radial scale. Four left-panel values are manually adjusted '
    r'for illustration.'
)

# ==================== EDIT 5: DRAWING CODE (ADVANCED CHANGES) ====================
# The entire drawing lives here. ${name} inserts a Python setting. You can
# add/remove/reorder TikZ elements below. No separate TeX source needs editing.
DOCUMENT = Template(r'''\documentclass{article}
\usepackage{fix-cm,graphicx,${font_packages},tikz}
\usetikzlibrary{calc}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tikzpicture}
\setlength{\PreviewBorder}{0pt}
\pagestyle{empty}
\begin{document}
\definecolor{MIPAAccent}{RGB}{${accent_rgb}}
\definecolor{MIPABarFill}{RGB}{${fill_rgb}}
\definecolor{MIPAText}{RGB}{${text_rgb}}
\definecolor{MIPAMuted}{RGB}{${muted_rgb}}
\definecolor{MIPAReal}{RGB}{${center_frame_rgb}}
\definecolor{MIPABackground}{RGB}{${background_rgb}}
\colorlet{MIPAGrid}{MIPAMuted!${grid_mix}!white}
\colorlet{MIPAFrame}{MIPAMuted!${frame_mix}!white}
\begin{tikzpicture}[x=1bp,y=1bp,inner sep=0pt,outer sep=0pt,
  every node/.style={font=${font_family}\upshape}]
  % Canvas and grid; drawing order controls what covers what.
  \path[use as bounding box] (-${half_page},-${half_page}) rectangle (${half_page},${half_page});
  \fill[MIPABackground] (-${half_page},-${half_page}) rectangle (${half_page},${half_page});
  ${grid_commands}
  \foreach \MIPAFile/\MIPAProbability/\MIPALabel/\MIPAAngle
    [count=\MIPAIndex from 0] in {${data}} {
    % Colored bars and connecting spokes.
    \pgfmathsetmacro{\MIPAStart}{\MIPAAngle-${bar_half_angle}}
    \pgfmathsetmacro{\MIPAEnd}{\MIPAAngle+${bar_half_angle}}
    \pgfmathsetmacro{\MIPAOuter}{${bar_inner}+${bar_length}*\MIPAProbability/${scale_max}}
    \draw[MIPAGrid,line width=${spoke_width}bp] (\MIPAAngle:${spoke_start}) -- (\MIPAAngle:${spoke_end});
    \path[fill=MIPABarFill,draw=MIPAAccent!${bar_outline_mix}!white,line width=${bar_line_width}bp,line join=round]
      (\MIPAStart:${bar_inner}) -- (\MIPAStart:\MIPAOuter)
      arc[start angle=\MIPAStart,end angle=\MIPAEnd,radius=\MIPAOuter bp]
      -- (\MIPAEnd:${bar_inner})
      arc[start angle=\MIPAEnd,end angle=\MIPAStart,radius=${bar_inner}bp] -- cycle;
    % Number placement, rotation, font, and content.
    \pgfmathsetmacro{\MIPALabelRadius}{\MIPAOuter+${label_gap}}
    \pgfmathsetmacro{\MIPALabelAngle}{\MIPAAngle+${label_rotation_offset}}
    \node[text=MIPAText,inner sep=${label_padding}bp,rotate=\MIPALabelAngle,anchor=${label_anchor},
      font=${font_family}${label_weight}${label_shape}\fontsize{${label_size}bp}{${label_line_height}bp}\selectfont]
      (MIPALabel-\MIPAIndex) at (\MIPAAngle:\MIPALabelRadius)
      {\MIPALabel${label_suffix}};
    % Actual typeset geometry for the optional layout checks.
    \path let \p1=(MIPALabel-\MIPAIndex.east), \p2=(MIPALabel-\MIPAIndex.west),
      \p3=(MIPALabel-\MIPAIndex.north), \p4=(MIPALabel-\MIPAIndex.south),
      \n1={veclen(\x1-\x2,\y1-\y2)}, \n2={veclen(\x3-\x4,\y3-\y4)}
      in \pgfextra{\typeout{MIPA_LABEL:\MIPAIndex:\n1:\n2:0pt:\MIPAOuter}
        \typeout{MIPA_BASE:\MIPAIndex:\x4:\y4:\x3:\y3}};
  }
  % Center circle, real image, and frame.
  \filldraw[fill=MIPABackground,draw=MIPAMuted!${circle_mix}!white,line width=${circle_width}bp]
    (0,0) circle[radius=${circle_radius}bp];
  \draw[MIPAReal,line width=${center_frame_width}bp] (-${center_frame},-${center_frame}) rectangle (${center_frame},${center_frame});
  \node at (0,0) {\includegraphics[width=${center_image_size}bp,height=${center_image_size}bp]{${center_path}}};
  % Surrounding synthetic images and their frames.
  \foreach \MIPAFile/\MIPAProbability/\MIPALabel/\MIPAAngle
    [count=\MIPAIndex from 0] in {${data}} {
    \colorlet{MIPABorder}{MIPAFrame}\def\MIPABorderWidth{${frame_width}bp}
    \ifdim\MIPAProbability pt>${highlight_threshold}pt
      \colorlet{MIPABorder}{MIPAAccent}\def\MIPABorderWidth{${highlight_frame_width}bp}
    \fi
    \begin{scope}[shift={(\MIPAAngle:${image_radius})}]
      \draw[MIPABorder,line width=\MIPABorderWidth] (-${image_frame},-${image_frame}) rectangle (${image_frame},${image_frame});
      \node at (0,0) {\includegraphics[width=${image_size}bp,height=${image_size}bp]{${image_dir}/\MIPAFile}};
    \end{scope}
  }
\end{tikzpicture}
\end{document}
''')


def panel_settings(panel):
    """Combine shared settings and panel overrides; derive image/bar positions."""
    settings = STYLE | panel.get('style', {})
    count = len(panel['values'])
    if not count or len(panel['images']) != count:
        raise ValueError(f"{panel['name']}: provide one image for each value")
    if any(not math.isfinite(v) or v < 0 for v in panel['values']):
        raise ValueError('Bar values must be finite and nonnegative')
    for key in ('page_size', 'image_size', 'center_image_size', 'scale_max', 'label_size'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    angles = panel.get('angles', [settings['start_angle']-360*i/count for i in range(count)])
    if len(angles) != count or any(not math.isfinite(a) for a in angles):
        raise ValueError('Provide one finite angle for each value')
    return settings, angles


def render_source(panel):
    """Translate the editable Python settings into a complete TeX document."""
    settings, angles = panel_settings(panel)
    image_dir = Path(panel['image_dir']).resolve()
    for filename in [panel['center_image'], *panel['images']]:
        if not (image_dir/filename).is_file():
            raise FileNotFoundError(image_dir/filename)
    data = []
    for filename, value, angle in zip(panel['images'], panel['values'], angles):
        # TikZ lists use commas and slashes; image_dir holds the directory path.
        if any(char in filename for char in '/,{}%\\'):
            raise ValueError('Use image basenames without TeX list separators; put their folder in image_dir')
        data.append(f"{filename}/{value:g}/{format(value, settings['number_format'])}/{angle:g}")
    replacements = settings | panel | {
        'half_page': settings['page_size']/2,
        'image_frame': settings['image_size']/2+settings['frame_padding'],
        'center_frame': settings['center_image_size']/2+settings['frame_padding'],
        'image_dir': str(image_dir),
        'center_path': str(image_dir/panel['center_image']),
        'data': ',\n    '.join(data),
        'grid_commands': '\n  '.join(
            rf'\draw[MIPAGrid,line width={settings["grid_width"]}bp,dash pattern=on '
            rf'{settings["grid_dash_on"]}bp off {settings["grid_dash_off"]}bp] '
            rf'(0,0) circle[radius={radius}bp];' for radius in settings['grid_radii']),
    }
    for key, value in replacements.copy().items():
        if key.endswith('_rgb'):
            if len(value) != 3 or any(not 0 <= channel <= 255 for channel in value):
                raise ValueError(f'{key}: use three RGB channels in [0, 255]')
            replacements[key] = ','.join(map(str, value))
    return '% Generated by generate_figures.py; edit the Python file.\n'+DOCUMENT.substitute(replacements)


# ==================== EDIT 6: OPTIONAL LAYOUT CHECKS ====================
# Checks follow your current settings, not fixed reference numbers/dimensions.
# CHECK_LAYOUT=False skips geometry checks for intentional layout changes.
def verify_outputs(output_dir=OUTPUT_DIR):
    for panel in PANELS:
        style, angles = panel_settings(panel)
        stem = Path(output_dir)/panel['name']
        info = subprocess.check_output(['pdfinfo', str(stem.with_suffix('.pdf'))], text=True)
        size = re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info)
        assert size and re.search(r'Pages:\s+1\b', info), f'{stem}: expected one page'
        assert all(abs(float(v)-style['page_size']) < .02 for v in size.groups()), 'Figure exceeds square page bounds'
        log = stem.with_suffix('.log').read_text()
        assert 'LaTeX Error' not in log and 'Overfull' not in log, f'Check {stem}.log'
        extracted = subprocess.check_output(['pdftotext', '-raw', str(stem.with_suffix('.pdf')), '-'], text=True)
        expected = ''.join(format(v, style['number_format'])+style['label_suffix'].replace(r'\%', '%') for v in panel['values'])
        assert re.sub(r'\s+', '', extracted) == re.sub(r'\s+', '', expected), 'PDF labels differ from Python values'
        if not CHECK_LAYOUT:
            continue
        labels = re.findall(r'MIPA_LABEL:(\d+):([\d.]+)pt:([\d.]+)pt:([\d.]+)pt:([\d.]+)', log)
        anchors = re.findall(r'MIPA_BASE:(\d+):(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt', log)
        assert len(labels) == len(anchors) == len(panel['values']), 'Missing label geometry'
        frame = style['image_size']/2+style['frame_padding']+max(style['frame_width'], style['highlight_frame_width'])/2
        image_boxes = []
        for angle in angles:
            x, y = (style['image_radius']*f(math.radians(angle)) for f in (math.cos, math.sin))
            image_boxes.append((x-frame, y-frame, x+frame, y+frame))
        center = style['center_image_size']/2+style['frame_padding']+style['center_frame_width']/2
        image_boxes.append((-center, -center, center, center))
        boxes = []
        for label, anchor in zip(labels, anchors):
            index, width, height, depth, radius = label
            assert index == anchor[0]
            index = int(index)
            angle = math.radians(angles[index])
            radius = float(radius)
            assert math.isclose(radius, style['bar_inner']+style['bar_length']*panel['values'][index]/style['scale_max'], abs_tol=.002)
            hw = float(width)*72/72.27/2
            hh = (float(height)+float(depth))*72/72.27/2
            bx, by, tx, ty = (float(v)*72/72.27 for v in anchor[1:])
            rotation = math.atan2(ty-by, tx-bx)-math.pi/2
            cx, cy = (bx+tx)/2, (by+ty)/2
            corners = [(cx+u*math.cos(rotation)-v*math.sin(rotation),
                        cy+u*math.sin(rotation)+v*math.cos(rotation))
                       for u in (-hw, hw) for v in (-hh, hh)]
            box = (min(x for x,y in corners), min(y for x,y in corners), max(x for x,y in corners), max(y for x,y in corners))
            relative_rotation = rotation-(angle-math.pi/2)
            projected_width = 2*(hw*abs(math.cos(relative_rotation))+hh*abs(math.sin(relative_rotation)))
            assert projected_width < 2*radius*math.sin(math.radians(style['bar_half_angle'])), f'{stem.name} label {index+1} is wider than its bar'
            for other in image_boxes+boxes:
                assert box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1], f'{stem.name} label {index+1} overlaps another element'
            boxes.append(box)
        for i, a in enumerate(image_boxes):
            for b in image_boxes[i+1:]:
                assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1], 'Images overlap'
        assert all(abs(v) < style['page_size']/2 for box in boxes+image_boxes for v in box), 'An image or number is clipped'
    print('PASS: PDFs match the editable Python settings and labels.'+(' No label/image overlaps or clipping.' if CHECK_LAYOUT else ' Geometry checks disabled.'))


# ==================== EDIT 7: EXPORT / BUILD STEPS ====================
def main():
    missing = [name for name in ('pdflatex', 'pdftoppm', 'pdfinfo', 'pdftotext') if shutil.which(name) is None]
    if missing:
        raise SystemExit('Missing required commands: '+', '.join(missing))
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prepare both panels before overwriting any generated files.
    sources = [(panel, render_source(panel)) for panel in PANELS]
    for panel, source in sources:
        name = panel['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Output names must use letters, numbers, underscores, or hyphens')
        (output_dir/f'{name}.tex').write_text(source)
        log = output_dir/f'{name}_compile.stdout'
        with log.open('w') as output:
            completed = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'{name}.tex'],
                cwd=output_dir, env=os.environ | {'max_print_line': '1000'}, stdout=output, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise SystemExit(f'LaTeX compilation failed; see {log}')
    verify_outputs(output_dir)
    for panel in PANELS:
        name = panel['name']
        subprocess.run(['pdftoppm', '-singlefile', '-scale-to', str(PREVIEW_PIXELS), '-png',
            f'{name}.pdf', f'{name}_preview'], cwd=output_dir, check=True)
        print(output_dir/f'{name}.pdf')
        print(output_dir/f'{name}_preview.png')
    includes = r'\hfill'.join(r'\includegraphics[width=.49\linewidth]{'+p['name']+'.pdf}' for p in PANELS)
    (output_dir/'intro.tex').write_text(
        '% Generated by generate_figures.py; edit CAPTION there.\n'
        '\\begin{figure*}[t]\n\\centering\n'+includes+'\n\\caption{'+CAPTION+'}\n'
        '\\label{fig:prototype-assignments}\n\\end{figure*}\n')


if __name__ == '__main__':
    main()
