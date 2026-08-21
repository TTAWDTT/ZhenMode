"""Generate a 4-slide condensed PPTX for a 1-2 minute presentation."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Colors ──────────────────────────────────────────────────────────
OCEAN_DEEP = RGBColor(0x0B, 0x1F, 0x3A)
OCEAN_MID  = RGBColor(0x12, 0x3A, 0x5E)
OCEAN_LITE = RGBColor(0x1E, 0x5A, 0x8C)
ACCENT_CY  = RGBColor(0x00, 0xBF, 0xD0)
ACCENT_OR  = RGBColor(0xF5, 0x9E, 0x0B)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xE2, 0xE8, 0xF0)
DARK_TEXT  = RGBColor(0x1E, 0x29, 0x3B)
GREEN_OK   = RGBColor(0x22, 0xC5, 0x55)
RED_FAIL   = RGBColor(0xEF, 0x44, 0x44)

W = prs.slide_width
H = prs.slide_height
FONT = 'Microsoft YaHei'

def add_bg(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color

def textbox(slide, left, top, width, height, text, size=14,
            color=DARK_TEXT, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = FONT
    p.alignment = align
    return tb

def bullets(slide, left, top, width, height, items, size=14,
            color=DARK_TEXT, spacing=6):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.name = FONT
        p.space_after = Pt(spacing)
    return tb

def rect(slide, left, top, width, height, fill_color, text="",
         size=12, font_color=WHITE, bold=False):
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
        p.font.size = Pt(size)
        p.font.color.rgb = font_color
        p.font.bold = bold
        p.font.name = FONT
        p.alignment = PP_ALIGN.CENTER
    return shape

# ════════════════════════════════════════════════════════════════════
# Slide 1: Title + Overview
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, OCEAN_DEEP)

textbox(slide, Inches(0.8), Inches(0.5), Inches(11), Inches(0.6),
        "谱方法海洋动力学求解器", size=32, color=WHITE, bold=True)
textbox(slide, Inches(0.8), Inches(1.1), Inches(11), Inches(0.45),
        "Spectral Ocean Model Solver  |  JAX + Strang Splitting",
        size=16, color=ACCENT_CY)

# Three key cards
card_w = Inches(3.5)
card_h = Inches(2.8)
gap = Inches(0.4)
start_x = Inches(0.8)
card_y = Inches(2.2)

# Card 1: What
rect(slide, start_x, card_y, card_w, Inches(0.45), OCEAN_MID,
     text="做什么", size=14, font_color=WHITE, bold=True)
bullets(slide, start_x + Inches(0.15), card_y + Inches(0.6),
        card_w - Inches(0.3), card_h - Inches(0.7),
        ["三维静力原始方程海洋模型",
         "谱方法 + Strang 分裂",
         "JAX 实现，自动微分",
         "自由表面 + 斜压耦合"],
        size=12, color=LIGHT_GRAY, spacing=8)

# Card 2: Tech stack
rect(slide, start_x + card_w + gap, card_y, card_w, Inches(0.45), OCEAN_MID,
     text="技术栈", size=14, font_color=WHITE, bold=True)
bullets(slide, start_x + card_w + gap + Inches(0.15), card_y + Inches(0.6),
        card_w - Inches(0.3), card_h - Inches(0.7),
        ["JAX 0.11 (CPU, x64)",
         "Python 3.14",
         "谱方法求解斜压模态",
         "Smagorinsky 混合",
         "UNESCO 状态方程"],
        size=12, color=LIGHT_GRAY, spacing=8)

# Card 3: Code stats
rect(slide, start_x + 2*(card_w + gap), card_y, card_w, Inches(0.45), OCEAN_MID,
     text="代码规模", size=14, font_color=WHITE, bold=True)
bullets(slide, start_x + 2*(card_w + gap) + Inches(0.15), card_y + Inches(0.6),
        card_w - Inches(0.3), card_h - Inches(0.7),
        ["核心求解器 ~780 行",
         "测试代码 ~1500 行",
         "脚本 ~750 行",
         "6 次提交，完整 git 记录"],
        size=12, color=LIGHT_GRAY, spacing=8)

# ════════════════════════════════════════════════════════════════════
# Slide 2: Architecture & Physics
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

textbox(slide, Inches(0.8), Inches(0.3), Inches(11), Inches(0.5),
        "架构与物理过程", size=28, color=OCEAN_DEEP, bold=True)

# Architecture flow
textbox(slide, Inches(0.8), Inches(1.0), Inches(5), Inches(0.35),
        "Strang 分裂时间推进", size=16, color=OCEAN_MID, bold=True)

# Flow boxes
flow_y = Inches(1.5)
box_w = Inches(2.2)
box_h = Inches(0.65)
bx = Inches(0.8)

rect(slide, bx, flow_y, box_w, box_h, OCEAN_LITE,
     text="斜压步 (显式)\nTracer + 3D u,v", size=11, bold=True)
rect(slide, bx + box_w + Inches(0.2), flow_y, box_w, box_h, OCEAN_MID,
     text="正压步 (谱方法)\nFree Surface + eta", size=11, bold=True)
rect(slide, bx + 2*(box_w + Inches(0.2)), flow_y, box_w, box_h, OCEAN_DEEP,
     text="物理参数化\nSmagorinsky + 摩擦", size=11, bold=True)

# Arrows
for i in range(2):
    ar = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
        bx + box_w + Inches(0.02) + i*(box_w + Inches(0.2)),
        flow_y + Inches(0.15), Inches(0.16), Inches(0.35))
    ar.fill.solid()
    ar.fill.fore_color.rgb = ACCENT_OR
    ar.line.fill.background()

# Key equations
textbox(slide, Inches(0.8), Inches(2.5), Inches(5), Inches(0.35),
        "关键方程", size=16, color=OCEAN_MID, bold=True)

bullets(slide, Inches(0.8), Inches(2.9), Inches(5.5), Inches(3.5),
        ["动量方程: du/dt = -1/rho * grad(p) + f*k x u + F",
         "连续性方程: dT/dt = -u . grad(T) + K * nabla^2 T",
         "自由表面: d(eta)/dt = -div(H * u_bar)",
         "状态方程: rho = UNESCO(T, S, p)",
         "底摩擦: tau_b = Cd * |u| * u"],
        size=13, color=DARK_TEXT, spacing=10)

# Right side: physics cards
textbox(slide, Inches(7.0), Inches(1.0), Inches(5.5), Inches(0.35),
        "已实现物理过程", size=16, color=OCEAN_MID, bold=True)

physics_items = [
    ("自由表面", "线性浅水方程，谱空间精确求解"),
    ("斜压耦合", "温度/盐度对流 + 压力梯度力"),
    ("Smagorinsky 混合", "亚网格尺度湍流参数化"),
    ("UNESCO EOS", "非线性状态方程，密度计算"),
    ("底摩擦", "二次底摩擦，Cd = 2.5e-3"),
    ("风强迫", "2D 风应力场驱动表面"),
]

py = Inches(1.5)
for title, desc in physics_items:
    rect(slide, Inches(7.0), py, Inches(5.5), Inches(0.45), LIGHT_GRAY,
         text="", size=11)
    textbox(slide, Inches(7.15), py + Inches(0.03), Inches(1.8), Inches(0.4),
            title, size=12, color=OCEAN_DEEP, bold=True)
    textbox(slide, Inches(9.0), py + Inches(0.03), Inches(3.4), Inches(0.4),
            desc, size=11, color=DARK_TEXT)
    py += Inches(0.52)

# ════════════════════════════════════════════════════════════════════
# Slide 3: Stability Problem
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, WHITE)

textbox(slide, Inches(0.8), Inches(0.3), Inches(11), Inches(0.5),
        "稳定性问题与诊断", size=28, color=OCEAN_DEEP, bold=True)

# Problem
rect(slide, Inches(0.8), Inches(1.0), Inches(11.5), Inches(0.5), RED_FAIL,
     text="问题: 模型在 ~50 步后出现 NaN (数值不稳定)", size=14,
     font_color=WHITE, bold=True)

# Diagnosis flow
textbox(slide, Inches(0.8), Inches(1.8), Inches(5), Inches(0.35),
        "诊断过程", size=16, color=OCEAN_MID, bold=True)

diag_items = [
    "1. 排除 Strang 分裂共振 (barotropic wind 已修复)",
    "2. 定位到斜压 PGF + tracer 耦合回路",
    "3. Forward Euler 在该回路中不稳定",
    "4. 半隐式 tracer-first 方案尝试失败",
    "   (tracer 平流仍显式依赖 u)",
]
bullets(slide, Inches(0.8), Inches(2.2), Inches(5.5), Inches(3),
        diag_items, size=13, color=DARK_TEXT, spacing=8)

# Root cause
rect(slide, Inches(0.8), Inches(4.5), Inches(5.5), Inches(0.5), ACCENT_OR,
     text="根因: 斜压 PGF-Tracer 耦合的显式时间推进",
     size=12, font_color=WHITE, bold=True)

bullets(slide, Inches(0.8), Inches(5.1), Inches(5.5), Inches(1.5),
        ["rho(T) -> PGF -> u -> T^new -> rho^new 形成正反馈",
         "显式方案无法抑制该反馈回路",
         "需要隐式或半隐式处理耦合项"],
        size=12, color=DARK_TEXT, spacing=6)

# Fix directions
textbox(slide, Inches(7.0), Inches(1.8), Inches(5.5), Inches(0.35),
        "下一步修复方向", size=16, color=OCEAN_MID, bold=True)

fixes = [
    ("RK2 / Predictor-Corrector", "中等复杂度，最可能有效", GREEN_OK),
    ("完全隐式斜压 PGF", "高复杂度，最严格", OCEAN_MID),
    ("垂直模态分解", "高复杂度，物理上最优雅", OCEAN_LITE),
]

fy = Inches(2.2)
for title, desc, color in fixes:
    rect(slide, Inches(7.0), fy, Inches(5.5), Inches(0.55), color,
         text=f"{title}\n{desc}", size=12, bold=True)
    fy += Inches(0.65)

# ════════════════════════════════════════════════════════════════════
# Slide 4: Summary
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, OCEAN_DEEP)

textbox(slide, Inches(0.8), Inches(0.5), Inches(11), Inches(0.5),
        "总结与现状", size=28, color=WHITE, bold=True)

# Status cards
card_w = Inches(3.5)
card_h = Inches(2.5)
gap = Inches(0.4)
start_x = Inches(0.8)
card_y = Inches(1.3)

# Done
rect(slide, start_x, card_y, card_w, Inches(0.45), GREEN_OK,
     text="已完成", size=14, bold=True)
bullets(slide, start_x + Inches(0.15), card_y + Inches(0.55),
        card_w - Inches(0.3), card_h - Inches(0.6),
        ["完整谱方法求解器",
         "自由表面 + 斜压耦合",
         "6 种物理过程参数化",
         "Strang 分裂时间推进",
         "完整测试套件"],
        size=12, color=LIGHT_GRAY, spacing=6)

# In progress
rect(slide, start_x + card_w + gap, card_y, card_w, Inches(0.45), ACCENT_OR,
     text="进行中", size=14, bold=True)
bullets(slide, start_x + card_w + gap + Inches(0.15), card_y + Inches(0.55),
        card_w - Inches(0.3), card_h - Inches(0.6),
        ["斜压 PGF 稳定性修复",
         "RK2 预测-校正方案",
         "(代码已就绪，待实现)",
         "修复后可进行长期积分"],
        size=12, color=LIGHT_GRAY, spacing=6)

# Next
rect(slide, start_x + 2*(card_w + gap), card_y, card_w, Inches(0.45), ACCENT_CY,
     text="下一步", size=14, bold=True)
bullets(slide, start_x + 2*(card_w + gap) + Inches(0.15), card_y + Inches(0.55),
        card_w - Inches(0.3), card_h - Inches(0.6),
        ["实现 RK2 时间推进",
         "通过稳定性测试",
         "1 天 / 30 天积分验证",
         "物理守恒量检验"],
        size=12, color=LIGHT_GRAY, spacing=6)

# Bottom line
rect(slide, Inches(0.8), Inches(4.5), Inches(11.5), Inches(0.6), OCEAN_MID,
     text="核心贡献: 基于 JAX 的谱方法海洋模型，完整物理参数化，开源可复现",
     size=14, font_color=WHITE, bold=True)

# Save
out = "ocean_solver_short.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
