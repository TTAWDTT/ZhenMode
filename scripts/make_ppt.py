"""Generate a project summary PPTX for the spectral ocean model."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import math

prs = Presentation()
prs.slide_width  = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)

# ── Color palette ───────────────────────────────────────────────────
OCEAN_DEEP = RGBColor(0x0B, 0x1F, 0x3A)   # deep navy
OCEAN_MID  = RGBColor(0x12, 0x3A, 0x5E)   # mid blue
OCEAN_LITE = RGBColor(0x1E, 0x5A, 0x8C)   # lighter blue
ACCENT_CY  = RGBColor(0x00, 0xBF, 0xD0)   # cyan accent
ACCENT_OR  = RGBColor(0xF5, 0x9E, 0x0B)   # orange accent
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xE2, 0xE8, 0xF0)
MID_GRAY   = RGBColor(0x94, 0xA3, 0xB8)
DARK_TEXT  = RGBColor(0x1E, 0x29, 0x3B)
GREEN_OK   = RGBColor(0x22, 0xC5, 0x55)
RED_FAIL   = RGBColor(0xEF, 0x44, 0x44)

W = prs.slide_width
H = prs.slide_height

def add_bg(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color

def add_textbox(slide, left, top, width, height, text, font_size=14,
                color=DARK_TEXT, bold=False, alignment=PP_ALIGN.LEFT,
                font_name='Microsoft YaHei'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_bullet_list(slide, left, top, width, height, items, font_size=14,
                    color=DARK_TEXT, font_name='Microsoft YaHei'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(6)
        p.level = 0
    return txBox

def add_rounded_rect(slide, left, top, width, height, fill_color, text="",
                     font_size=12, font_color=WHITE, bold=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color
        p.font.bold = bold
        p.font.name = 'Microsoft YaHei'
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].space_before = Pt(0)
        tf.paragraphs[0].space_after = Pt(0)
    return shape

def add_card(slide, left, top, width, height, title, body_lines,
             title_color=OCEAN_MID, body_color=DARK_TEXT,
             title_size=14, body_size=11, fill_color=LIGHT_GRAY):
    """Add a card with title bar + body."""
    # Background
    bg = add_rounded_rect(slide, left, top, width, height, fill_color)
    bg.fill.fore_color.rgb = fill_color
    # Title bar
    title_h = Inches(0.38)
    add_rounded_rect(slide, left, top, width, title_h, title_color,
                     text=title, font_size=title_size, font_color=WHITE, bold=True)
    # Body
    body_top = top + title_h + Inches(0.08)
    body_h = height - title_h - Inches(0.16)
    txBox = slide.shapes.add_textbox(left + Inches(0.15), body_top,
                                     width - Inches(0.30), body_h)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(body_lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(body_size)
        p.font.color.rgb = body_color
        p.font.name = 'Microsoft YaHei'
        p.space_after = Pt(4)
    return bg

# ══════════════════════════════════════════════════════════════════════
# Slide 1: Title
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, OCEAN_DEEP)

# Decorative wave SVG
wave_svg = '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1333 750">
  <path d="M0,600 Q200,580 400,600 T800,600 T1200,600 L1333,600 L1333,750 L0,750 Z"
        fill="#123A5E" opacity="0.6"/>
  <path d="M0,640 Q200,620 400,640 T800,640 T1200,640 L1333,640 L1333,750 L0,750 Z"
        fill="#1E5A8C" opacity="0.4"/>
  <path d="M0,680 Q200,660 400,680 T800,680 T1200,680 L1333,680 L1333,750 L0,750 Z"
        fill="#00BFD0" opacity="0.2"/>
</svg>
'''
import tempfile, os
svg_path = os.path.join(tempfile.gettempdir(), 'wave_bg.svg')
with open(svg_path, 'w') as f:
    f.write(wave_svg)

# Title
add_textbox(slide, Inches(1.0), Inches(1.8), Inches(11.3), Inches(1.2),
            "谱方法海洋模型", font_size=48, color=WHITE, bold=True,
            alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1.0), Inches(3.0), Inches(11.3), Inches(0.8),
            "Spectral-Method Ocean Model", font_size=24, color=ACCENT_CY,
            bold=False, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1.0), Inches(4.2), Inches(11.3), Inches(0.6),
            "静力原始方程  |  JAX JIT 加速  |  伪谱方法", font_size=18,
            color=MID_GRAY, alignment=PP_ALIGN.CENTER)
add_textbox(slide, Inches(1.0), Inches(5.8), Inches(11.3), Inches(0.5),
            "2026-08-21", font_size=14, color=MID_GRAY,
            alignment=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════════
# Slide 2: What does this model do?
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "模型做什么？", font_size=32, color=OCEAN_DEEP, bold=True)
add_textbox(slide, Inches(0.8), Inches(1.0), Inches(11.7), Inches(0.5),
            "从零构建的三维海洋数值模型，求解静力原始方程", font_size=16,
            color=MID_GRAY)

# Left column: equation
add_textbox(slide, Inches(0.8), Inches(1.8), Inches(5.5), Inches(0.4),
            "控制方程", font_size=16, color=OCEAN_MID, bold=True)

eq_items = [
    "∂u/∂t + u∂u/∂x + v∂u/∂y = fv - (1/ρ₀)∂p/∂x + ν∇²u + F_wind",
    "∂v/∂t + u∂v/∂x + v∂v/∂y = -fu - (1/ρ₀)∂p/∂y + ν∇²v",
    "∂p/∂z = -ρg                          (静力平衡)",
    "∂T/∂t + u∇T = κ∇²T + Q_heat",
    "∂η/∂t + ∇·(H·u_bt) = 0          (连续性/自由表面)",
]
txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(6.0), Inches(3.0))
tf = txBox.text_frame
tf.word_wrap = True
for i, eq in enumerate(eq_items):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = eq
    p.font.size = Pt(12)
    p.font.color.rgb = OCEAN_MID
    p.font.name = 'Consolas'
    p.space_after = Pt(8)

# Right column: key facts
add_textbox(slide, Inches(7.2), Inches(1.8), Inches(5.3), Inches(0.4),
            "关键参数", font_size=16, color=OCEAN_MID, bold=True)

facts = [
    ("方程", "静力原始方程 (HPE)"),
    ("水平离散", "伪谱方法 (FFT)"),
    ("垂直离散", "有限差分 (z 坐标)"),
    ("时间推进", "IMEX Strang 分裂"),
    ("网格", "128×128×14 (0.1°, ~9km)"),
    ("计算后端", "JAX JIT (CPU)"),
    ("加速比", "12.4x vs numpy/scipy"),
    ("精度", "max diff 2.8e-13"),
]
for i, (k, v) in enumerate(facts):
    y = Inches(2.3 + i * 0.45)
    add_textbox(slide, Inches(7.2), y, Inches(1.8), Inches(0.35),
                k, font_size=12, color=MID_GRAY)
    add_textbox(slide, Inches(9.2), y, Inches(3.3), Inches(0.35),
                v, font_size=12, color=DARK_TEXT, bold=True)

# Bottom: why spectral?
add_rounded_rect(slide, Inches(0.8), Inches(5.5), Inches(11.7), Inches(1.4),
                 LIGHT_GRAY)
add_textbox(slide, Inches(1.0), Inches(5.6), Inches(11.3), Inches(0.4),
            "为什么用谱方法？", font_size=14, color=OCEAN_MID, bold=True)
add_textbox(slide, Inches(1.0), Inches(6.0), Inches(11.3), Inches(0.8),
            "物理空间求导 = 看邻居 (有限差分 stencil, 缓存不友好)\n"
            "傅里叶空间求导 = 每个频率乘一个常数 ik (完全独立, 无邻居依赖)\n"
            "结果：求导变成对角矩阵乘法，精度达谱收敛 (指数收敛)",
            font_size=12, color=DARK_TEXT)

# ══════════════════════════════════════════════════════════════════════
# Slide 3: Project structure & code stats
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "项目结构与代码规模", font_size=32, color=OCEAN_DEEP, bold=True)

# Code stats cards
stats = [
    ("~3,300", "行核心代码 (src/)"),
    ("~1,500", "行测试代码 (tests/)"),
    ("~750", "行脚本 (scripts/)"),
    ("6", "次提交"),
]
for i, (num, desc) in enumerate(stats):
    x = Inches(0.8 + i * 3.0)
    add_rounded_rect(slide, x, Inches(1.3), Inches(2.7), Inches(1.1),
                     OCEAN_MID)
    add_textbox(slide, x, Inches(1.45), Inches(2.7), Inches(0.5),
                num, font_size=28, color=ACCENT_CY, bold=True,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x, Inches(1.95), Inches(2.7), Inches(0.4),
                desc, font_size=11, color=WHITE, alignment=PP_ALIGN.CENTER)

# Module table
add_textbox(slide, Inches(0.8), Inches(2.8), Inches(11.7), Inches(0.4),
            "核心模块", font_size=16, color=OCEAN_MID, bold=True)

modules = [
    ("jax_solver.py", "776", "JAX JIT 求解器 (主力)"),
    ("spectral_ops.py", "526", "FFT 算子 (导数/散度/拉普拉斯/泊松)"),
    ("integrator.py", "293", "numpy IMEX Strang 分裂"),
    ("grid.py", "218", "网格 + ETOPO 地形读取"),
    ("eos.py", "204", "线性 + UNESCO 状态方程"),
    ("forcing.py", "174", "风应力 + 热通量场生成"),
    ("momentum.py", "165", "动量方程右端项"),
    ("config.py", "182", "GridConfig + PhysicsConfig"),
]
for i, (name, lines, desc) in enumerate(modules):
    y = Inches(3.3 + i * 0.4)
    add_textbox(slide, Inches(0.8), y, Inches(2.5), Inches(0.35),
                name, font_size=11, color=OCEAN_MID, bold=True,
                font_name='Consolas')
    add_textbox(slide, Inches(3.5), y, Inches(0.8), Inches(0.35),
                lines, font_size=11, color=MID_GRAY,
                font_name='Consolas', alignment=PP_ALIGN.RIGHT)
    add_textbox(slide, Inches(4.5), y, Inches(7.5), Inches(0.35),
                desc, font_size=11, color=DARK_TEXT)

# Right side: test/verify info
add_textbox(slide, Inches(8.5), Inches(2.8), Inches(4.0), Inches(0.4),
            "验证体系", font_size=16, color=OCEAN_MID, bold=True)

verifies = [
    "test_wave_speed.py — 波速 c=√(gH) 验证",
    "test_forcing.py — Ekman 输运 + Sverdrup",
    "compare_jax_numpy.py — JAX vs numpy 一致性",
    "verify_stability.py — 稳定性测试 (3 cases)",
    "4 个 pytest 模块 (~1487 行)",
]
for i, v in enumerate(verifies):
    y = Inches(3.3 + i * 0.4)
    add_textbox(slide, Inches(8.5), y, Inches(4.0), Inches(0.35),
                "• " + v, font_size=11, color=DARK_TEXT)

# ══════════════════════════════════════════════════════════════════════
# Slide 4: Commit history timeline
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "开发历程：6 次提交", font_size=32, color=OCEAN_DEEP, bold=True)

# Timeline
commits = [
    ("1", "9dd0a66", "初始提交", "谱方法求解器 + ETOPO 地形 + CDO 重映射", "+3,796", OCEAN_DEEP),
    ("2", "b74d394", "向量化", "向量化循环 + 多线程 scipy.fft", "+132", OCEAN_MID),
    ("3", "c9984ad", "JAX 加速", "JAX JIT 求解器, 12.4x speedup", "+618", ACCENT_CY),
    ("4", "effe5ce", "自由表面", "H_sw=sum(dz), 精确浅水步", "+400", OCEAN_LITE),
    ("5", "855d98c", "物理参数化", "UNESCO EOS, Smagorinsky, 2D 强迫", "+745", ACCENT_OR),
    ("6", "bf0134d", "稳定性修复", "半隐式 PGF + 正压风移入线性步", "+739", RED_FAIL),
]

# Draw timeline line
line_y = Inches(4.5)
line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                               Inches(0.8), line_y,
                               Inches(11.7), Inches(0.04))
line.fill.solid()
line.fill.fore_color.rgb = OCEAN_MID
line.line.fill.background()

for i, (num, hash_, title, desc, lines, color) in enumerate(commits):
    x = Inches(0.8 + i * 2.0)
    # Circle on timeline
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                    x + Inches(0.65), line_y - Inches(0.13),
                                    Inches(0.3), Inches(0.3))
    circle.fill.solid()
    circle.fill.fore_color.rgb = color
    circle.line.fill.background()
    add_textbox(slide, x + Inches(0.65), line_y - Inches(0.10), Inches(0.3), Inches(0.25),
                num, font_size=10, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    # Alternate above/below
    if i % 2 == 0:
        # Above
        card_top = Inches(1.3)
        card_h = Inches(2.8)
    else:
        # Below
        card_top = Inches(5.0)
        card_h = Inches(2.2)

    add_rounded_rect(slide, x, card_top, Inches(1.9), card_h, LIGHT_GRAY)
    # Color bar
    add_rounded_rect(slide, x, card_top, Inches(1.9), Inches(0.32), color)
    add_textbox(slide, x, card_top + Inches(0.02), Inches(1.9), Inches(0.28),
                title, font_size=11, color=WHITE, bold=True,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.1), card_top + Inches(0.42), Inches(1.7), Inches(0.25),
                hash_, font_size=9, color=MID_GRAY, font_name='Consolas')
    add_textbox(slide, x + Inches(0.1), card_top + Inches(0.70), Inches(1.7), Inches(1.0),
                desc, font_size=9, color=DARK_TEXT)
    add_textbox(slide, x + Inches(0.1), card_top + Inches(1.75), Inches(1.7), Inches(0.3),
                lines + " lines", font_size=10, color=color, bold=True)

# ══════════════════════════════════════════════════════════════════════
# Slide 5: Architecture diagram
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "数值方法架构", font_size=32, color=OCEAN_DEEP, bold=True)

# Strang splitting diagram
add_textbox(slide, Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.4),
            "IMEX Strang 分裂: L(dt/2) → N(dt) → L(dt/2)", font_size=16,
            color=OCEAN_MID, bold=True)

# Three boxes for the splitting
box_y = Inches(1.8)
box_h = Inches(1.8)

# L(dt/2)
add_rounded_rect(slide, Inches(0.8), box_y, Inches(3.5), box_h, OCEAN_MID)
add_textbox(slide, Inches(0.9), box_y + Inches(0.1), Inches(3.3), Inches(0.35),
            "线性半步 L(dt/2)", font_size=14, color=WHITE, bold=True)
lin_items = [
    "• 水平扩散 (谱衰减, 精确)",
    "• f-平面科氏力 (精确旋转)",
    "• 自由表面 (精确浅水解)",
]
for i, item in enumerate(lin_items):
    add_textbox(slide, Inches(0.9), box_y + Inches(0.5 + i * 0.35),
                Inches(3.3), Inches(0.3), item, font_size=11, color=LIGHT_GRAY)

# Arrow
arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                               Inches(4.4), box_y + Inches(0.6),
                               Inches(0.5), Inches(0.5))
arrow.fill.solid()
arrow.fill.fore_color.rgb = ACCENT_OR
arrow.line.fill.background()

# N(dt)
add_rounded_rect(slide, Inches(5.0), box_y, Inches(3.5), box_h, ACCENT_OR)
add_textbox(slide, Inches(5.1), box_y + Inches(0.1), Inches(3.3), Inches(0.35),
            "非线性全步 N(dt)", font_size=14, color=WHITE, bold=True)
nl_items = [
    "• 平流 (通量形式, dealias)",
    "• 斜压压力梯度",
    "• 垂直扩散 (有限差分)",
    "• 风应力 + 热通量",
    "• 底摩擦 + Smagorinsky",
]
for i, item in enumerate(nl_items):
    add_textbox(slide, Inches(5.1), box_y + Inches(0.5 + i * 0.28),
                Inches(3.3), Inches(0.25), item, font_size=10, color=WHITE)

# Arrow 2
arrow2 = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                Inches(8.6), box_y + Inches(0.6),
                                Inches(0.5), Inches(0.5))
arrow2.fill.solid()
arrow2.fill.fore_color.rgb = ACCENT_OR
arrow2.line.fill.background()

# L(dt/2)
add_rounded_rect(slide, Inches(9.2), box_y, Inches(3.5), box_h, OCEAN_MID)
add_textbox(slide, Inches(9.3), box_y + Inches(0.1), Inches(3.3), Inches(0.35),
            "线性半步 L(dt/2)", font_size=14, color=WHITE, bold=True)
for i, item in enumerate(lin_items):
    add_textbox(slide, Inches(9.3), box_y + Inches(0.5 + i * 0.35),
                Inches(3.3), Inches(0.3), item, font_size=11, color=LIGHT_GRAY)

# Bottom: spectral method flow
add_textbox(slide, Inches(0.8), Inches(4.0), Inches(11.7), Inches(0.4),
            "谱方法求导流程", font_size=16, color=OCEAN_MID, bold=True)

flow_y = Inches(4.5)
flow_items = [
    ("物理空间 u\n(N 个点)", OCEAN_DEEP),
    ("FFT", OCEAN_MID),
    ("傅里叶空间 û\n(N 个频率)", OCEAN_LITE),
    ("× ik\n(逐元素乘法)", ACCENT_OR),
    ("IFFT", OCEAN_MID),
    ("物理空间 ∂u/∂x\n(N 个点)", OCEAN_DEEP),
]
for i, (label, color) in enumerate(flow_items):
    x = Inches(0.8 + i * 2.0)
    add_rounded_rect(slide, x, flow_y, Inches(1.7), Inches(1.0), color)
    add_textbox(slide, x, flow_y + Inches(0.15), Inches(1.7), Inches(0.7),
                label, font_size=11, color=WHITE, bold=True,
                alignment=PP_ALIGN.CENTER)
    if i < len(flow_items) - 1:
        arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                     x + Inches(1.75), flow_y + Inches(0.3),
                                     Inches(0.25), Inches(0.4))
        arr.fill.solid()
        arr.fill.fore_color.rgb = MID_GRAY
        arr.line.fill.background()

add_textbox(slide, Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0),
            "核心优势：傅里叶空间中求导 = 对角矩阵乘法 (每个频率独立), 无邻居耦合\n"
            "精度：谱收敛 (指数收敛), 远超有限差分的 O(Δx²)\n"
            "速度：FFT 是 O(N log N), 且各频率完全并行",
            font_size=12, color=DARK_TEXT)

# ═══════════════════════════用了 5 cards
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "物理过程 I：自由表面 + UNESCO 状态方程", font_size=28,
            color=OCEAN_DEEP, bold=True)

# Card 1: Free surface
add_card(slide, Inches(0.8), Inches(1.3), Inches(5.7), Inches(2.8),
         "自由表面 (Commit 4)",
         ["原来：刚盖近似 (η≡0), 海面是铁板",
          "  → 无表面重力波, 无风暴潮",
          "  → 无罗斯贝波, 无法与卫星高度计同化",
          "",
          "现在：η 随时间演化",
          "  → 谱空间精确求解浅水方程 (矩阵指数)",
          "  → 连续性: ∂η/∂t = -H·∇·u_bt",
          "  → 动量: ∂u_bt/∂t = -g·∇η",
          "",
          "验证: 波速误差 4.91%, 质量守恒 7e-15"],
         title_color=OCEAN_MID, body_size=11)

# Card 2: UNESCO EOS
add_card(slide, Inches(6.8), Inches(1.3), Inches(5.7), Inches(2.8),
         "UNESCO 非线性状态方程 (Commit 5)",
         ["原来：线性 ρ = ρ₀(1-αΔT+βΔS)",
          "  → 跨 13° 纬度误差 5-15%",
          "  → 无法模拟 cabbeling (密度对流)",
          "",
          "现在：UNESCO 1980 完整多项式",
          "  ρ = ρ_smow(T) + B(T)·S + C(T)·S^1.5 + D·S²",
          "  → 温度对密度的影响随 T 本身变化",
          "  → 冷水对温度更敏感 (非线性响应)",
          "",
          "切换: eos_type='unesco'"],
         title_color=OCEAN_LITE, body_size=11)

# Bottom row
add_card(slide, Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.8),
         "H_sw = sum(dz) 深度修复 (Commit 4)",
         ["Bug: 浅水频率用 H_mean=5724m",
          "  但 z 网格只覆盖 sum(dz)=4000m",
          "  → 20% 波速误差",
          "",
          "修复: H_sw = sum(dz) 作为有效深度",
          "  → dz_norm 归一化 (sum=1)",
          "  → 正压速度 = 真深度平均",
          "  → 提取-演化-投影 往返一致"],
         title_color=OCEAN_DEEP, body_size=11)

add_card(slide, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.8),
         "正压风移入线性步 (Commit 6)",
         ["问题: Strang 分裂把风 (显式) 和自由",
          "  表面 (线性) 分开 → 共振, η 爆炸",
          "  → 21 步内 η: 0 → 2.2×10⁴",
          "",
          "修复: 正压风 F=τ/(ρ₀H_sw) 加入线性步",
          "  → 求解强迫浅水方程精确解",
          "  → 显式步只保留斜压风分量",
          "  → 无条件稳定"],
         title_color=ACCENT_OR, body_size=11)

# ══════════════════════════════════════════════════════════════════════
# Slide 7: Physics II — Smagorinsky, 2D forcing, bottom friction
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "物理过程 II：次网格闭合 + 2D 强迫 + 底摩擦", font_size=28,
            color=OCEAN_DEEP, bold=True)

# Card 1: Smagorinsky
add_card(slide, Inches(0.8), Inches(1.3), Inches(3.7), Inches(5.5),
         "Smagorinsky\n次网格闭合",
         ["原来：常数黏性 ν_h=100",
          "  → 平静区耗散过多",
          "  → 强流区耗散不够",
          "  → 谱方法 Gibbs 振荡",
          "",
          "现在：ν_smg = (Cs·Δx)²·|D|",
          "  |D| = 变形率 = √((∂u/∂x-∂v/∂y)²",
          "         + (∂u/∂y+∂v/∂x)²)",
          "",
          "效果：",
          "  • 强流处黏性自动增大",
          "  • 弱流处黏性自动减小",
          "  • 模拟亚网格涡旋耗散",
          "  • 自适应抑制 Gibbs 振荡",
          "",
          "默认 Cs=0 (关闭)"],
         title_color=OCEAN_MID, body_size=10)

# Card 2: 2D forcing
add_card(slide, Inches(4.7), Inches(1.3), Inches(3.7), Inches(5.5),
         "二维空间\n强迫场",
         ["原来：全局标量, 无空间结构",
          "  → 无风应力旋度",
          "  → 不产生副热带回旋",
          "",
          "现在：2D (nx, ny) 场",
          "",
          "风应力 (Stommel 回旋):",
          "  τx(y) = -τ₀·cos(πy/Ly)",
          "  南: 东风 (贸易风)",
          "  北: 西风 (西风带)",
          "  中: 零应力",
          "",
          "热通量 (经向梯度):",
          "  Q(y) = -Q₀·(2y'-1)",
          "  南: 加热  北: 冷却",
          "",
          "意义：驱动真实大尺度环流"],
         title_color=ACCENT_OR, body_size=10)

# Card 3: Bottom friction
add_card(slide, Inches(8.6), Inches(1.3), Inches(3.7), Inches(5.5),
         "二次底摩擦\n(Quadratic Drag)",
         ["原来：线性 -r·u",
          "  → 高流速下阻尼不足",
          "  → 低估潮流耗散",
          "",
          "现在：τ = -Cd·|U|·u",
          "  |U| = √(u²+v²)",
          "  Cd = 0.0025 (标准值)",
          "",
          "效果：",
          "  • 高流速时阻尼 ~u² (更强)",
          "  • 正确的潮流能量耗散",
          "  • 更真实的底层边界层",
          "  • 对数边界层结构",
          "",
          "切换: bottom_friction=",
          "       'quadratic'"],
         title_color=OCEAN_DEEP, body_size=10)

# ══════════════════════════════════════════════════════════════════════
# Slide 8: Stability problem
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "稳定性问题：诊断与修复", font_size=32, color=OCEAN_DEEP, bold=True)

# Problem
add_rounded_rect(slide, Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.8),
                 RGBColor(0xFE, 0xE2, 0xE2))
add_textbox(slide, Inches(1.0), Inches(1.3), Inches(11.3), Inches(0.6),
            "问题：dt=300s 时数值发散 (NaN), 无论有无强迫 — 不是 CFL 问题 (内部波 CFL ~0.003)",
            font_size=14, color=RED_FAIL, bold=True)

# Diagnosis flow
add_textbox(slide, Inches(0.8), Inches(2.3), Inches(11.7), Inches(0.4),
            "诊断过程：逐项隔离", font_size=16, color=OCEAN_MID, bold=True)

diag_items = [
    ("关闭斜压 PGF", "→ 稳定 ✓", GREEN_OK),
    ("关闭 tracer 平流", "→ 稳定 ✓", GREEN_OK),
    ("两者同时开启", "→ 复现发散 ✗", RED_FAIL),
]
for i, (test, result, color) in enumerate(diag_items):
    y = Inches(2.8 + i * 0.45)
    add_textbox(slide, Inches(1.0), y, Inches(4.0), Inches(0.35),
                test, font_size=12, color=DARK_TEXT)
    add_textbox(slide, Inches(5.2), y, Inches(3.0), Inches(0.35),
                result, font_size=12, color=color, bold=True)

# Root cause
add_rounded_rect(slide, Inches(0.8), Inches(4.3), Inches(11.7), Inches(1.2),
                 RGBColor(0xFF, 0xF7, 0xED))
add_textbox(slide, Inches(1.0), Inches(4.4), Inches(11.3), Inches(0.4),
            "根因：Forward Euler 对斜压 PGF–tracer 耦合的本质不稳定", font_size=14,
            color=ACCENT_OR, bold=True)
add_textbox(slide, Inches(1.0), Inches(4.8), Inches(11.3), Inches(0.6),
            "u → 平流 T → T 异常 → 密度异常 → 斜压 PGF → 加速 u → ...\n"
            "Forward Euler 对振荡系统 |λ| > 1 (任何 dt > 0), 不是 CFL 违反",
            font_size=12, color=DARK_TEXT)

# Fix attempts
add_textbox(slide, Inches(0.8), Inches(5.8), Inches(11.7), Inches(0.4),
            "修复尝试", font_size=16, color=OCEAN_MID, bold=True)

# Fix 1: success
add_rounded_rect(slide, Inches(0.8), Inches(6.3), Inches(5.7), Inches(0.9),
                 RGBColor(0xE0, 0xF2, 0xFE))
add_textbox(slide, Inches(1.0), Inches(6.35), Inches(5.3), Inches(0.3),
            "✓ 正压风移入线性步 (成功)", font_size=12, color=OCEAN_MID, bold=True)
add_textbox(slide, Inches(1.0), Inches(6.65), Inches(5.3), Inches(0.5),
            "风+自由表面精确耦合, 消除 Strang 分裂共振",
            font_size=10, color=DARK_TEXT)

# Fix 2: fail
add_rounded_rect(slide, Inches(6.8), Inches(6.3), Inches(5.7), Inches(0.9),
                 RGBColor(0xFE, 0xE2, 0xE2))
add_textbox(slide, Inches(7.0), Inches(6.35), Inches(5.3), Inches(0.3),
            "✗ 半隐式斜压 PGF (失败)", font_size=12, color=RED_FAIL, bold=True)
add_textbox(slide, Inches(7.0), Inches(6.65), Inches(5.3), Inches(0.5),
            "tracer-first: PGF 用新 T/S, 但平流仍显式 → 3/3 测试 NaN",
            font_size=10, color=DARK_TEXT)

# ══════════════════════════════════════════════════════════════════════
# Slide 9: Stability fix directions
# ══════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)
add_textbox(slide, Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.6),
            "下一步：彻底修复方向", font_size=32, color=OCEAN_DEEP, bold=True)

add_textbox(slide, Inches(0.8), Inches(1.1), Inches(11.7), Inches(0.5),
            "半隐式修复失败的原因：PGF 对 T 半隐式了, 但 tracer 平流对 u 仍是显式的",
            font_size=14, color=MID_GRAY)

# Three options
options = [
    ("方案 A：垂直模态分解",
     "高复杂度",
     "将斜压 PGF 移入线性步\n按垂直正规模在谱空间精确求解\n\n优点: 最严谨, 无条件稳定\n缺点: 需推导垂直模态,\n       代码重构量大",
     OCEAN_DEEP),
    ("方案 B：RK2 预测-校正",
     "中等复杂度",
     "预测步: u*, T* = Euler\n校正步: 用平均值更新\n\n优点: 2阶精度, 稳定域大\n缺点: 每步计算量 ×2",
     ACCENT_OR),
    ("方案 C：Crank-Nicolson",
     "低复杂度",
     "PGF = 0.5×(PGF_old + PGF_new)\n需预测 T* 来算 PGF_new\n\n优点: 实现简单\n缺点: 需要预测步,\n       稳定性待验证",
     ACCENT_CY),
]
for i, (title, complexity, body, color) in enumerate(options):
    x = Inches(0.8 + i * 4.0)
    add_rounded_rect(slide, x, Inches(2.0), Inches(3.7), Inches(4.5), LIGHT_GRAY)
    add_rounded_rect(slide, x, Inches(2.0), Inches(3.7), Inches(0.5), color)
    add_textbox(slide, x, Inches(2.05), Inches(3.7), Inches(0.4),
                title, font_size=14, color=WHITE, bold=True,
                alignment=PP_ALIGN.CENTER)
    # complexity badge
    add_rounded_rect(slide, x + Inches(0.8), Inches(2.6), Inches(2.1), Inches(0.3),
                     color)
    add_textbox(slide, x + Inches(0.8), Inches(2.62), Inches(2.1), Inches(0.28),
                complexity, font_size=10, color=WHITE, bold=True,
                alignment=PP_ALIGN.CENTER)
    # body
    txBox = slide.shapes.add_textbox(x + Inches(0.15), Inches(3.1),
                                     Inches(3.4), Inches(3.2))
    tf = txBox.text_frame
    tf.word_wrap = True
    for j, line in enumerate(body.split('\n')):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.font.name = 'Microsoft YaHei'
        p.space_after = Pt(3)

# ══════════════════════════════════════════════════════════════════════
# Slide 10: Summary
# ═════════════════════════════════════)
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, OCEAN_DEEP)
add_textbox(slide, Inches(0.8), Inches(0.5), Inches(11.7), Inches(0.6),
            "总结", font_size=32, color=WHITE, bold=True)

# Achievement cards
ach = [
    ("从零构建", "6 次提交, ~5900 行代码\n从方程到可运行模型"),
    ("谱方法", "FFT 求导, 指数收敛\n比有限差更精确更高效"),
    ("JAX 加速", "12.4x vs numpy/scipy\n精度差 2.8e-13"),
    ("物理完整", "自由表面 + UNESCO EOS\n2D 强迫 + Smagorinsky"),
]
for i, (title, desc) in enumerate(ach):
    x = Inches(0.8 + i * 3.0)
    add_rounded_rect(slide, x, Inches(1.5), Inches(2.7), Inches(1.8),
                     OCEAN_MID)
    add_textbox(slide, x, Inches(1.65), Inches(2.7), Inches(0.4),
                title, font_size=16, color=ACCENT_CY, bold=True,
                alignment=PP_ALIGN.CENTER)
    add_textbox(slide, x + Inches(0.1), Inches(2.1), Inches(2.5), Inches(1.1),
                desc, font_size=11, color=LIGHT_GRAY,
                alignment=PP_ALIGN.CENTER)

# Current status
add_rounded_rect(slide, Inches(0.8), Inches(3.7), Inches(11.7), Inches(1.5),
                 OCEAN_MID)
add_textbox(slide, Inches(1.0), Inches(3.8), Inches(11.3), Inches(0.4),
            "当前状态", font_size=16, color=ACCENT_CY, bold=True)
add_textbox(slide, Inches(1.0), Inches(4.2), Inches(11.3), Inches(0.9),
            "✓ 正压风修复成功 (移入线性步, 精确强迫浅水解)\n"
            "✗ 斜压 PGF 修复失败 (半隐式不足, 平流仍显式 → 3/3 测试 NaN)\n"
            "→ 下一步: 垂直模态分解 / RK2 / Crank-Nicolson",
            font_size=13, color=WHITE)

# Key takeaway
add_textbox(slide, Inches(0.8), Inches(5.6), Inches(11.7), Inches(0.5),
            "核心洞察", font_size=18, color=ACCENT_CY, bold=True)
add_textbox(slide, Inches(0.8), Inches(6.1), Inches(11.7), Inches(1.0),
            "Forward Euler 对振荡耦合系统是本质不稳定的 (|λ|>1, 任何 dt)\n"
            "彻底解决需要: 将斜压 PGF 移入精确线性步, 或改用高阶时间格式",
            font_size=14, color=WHITE)

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
output_path = r'C:\Users\zhen.luo\ocean_solver\ocean_solver_summary.pptx'
prs.save(output_path)
print(f"PPTX saved to: {output_path}")
print(f"Slides: {len(prs.slides)}")
