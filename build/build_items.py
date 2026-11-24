"""Build the Kumo Supply test items (images, PDFs, videos) from code.

Usage: python3 build_items.py <output_dir>
Needs: playwright + Chrome, opencv-python, numpy, Pillow, ffmpeg.
Everything is generated from templates in this file. No generative AI is used,
so the answers are known by construction (see ../labels.yaml).
"""
import sys, os, subprocess, shutil, tempfile
import numpy as np, cv2
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "../kumo")
CHROME = shutil.which("google-chrome") or shutil.which("chromium")
rng = np.random.default_rng(42)

BASE_CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Noto Sans CJK JP', 'Noto Sans CJK TC', 'Helvetica', sans-serif; color: #222; }
"""

# ---------------------------------------------------------------- images ---
RECEIPT = """
<html><head><style>""" + BASE_CSS + """
body { background: transparent; }
.r { width: 440px; background: #fbfaf6; padding: 26px 24px 40px; font-family: 'Noto Sans CJK JP','Noto Sans CJK TC', monospace; font-size: 15px; line-height: 1.55; }
.c { text-align: center; } .b { font-weight: 700; font-size: 20px; }
.row { display: flex; justify-content: space-between; white-space: nowrap; gap: 16px; } hr { border: none; border-top: 1px dashed #888; }
.t { font-size: 22px; font-weight: 700; }
</style></head><body><div class="r" id="cap">
<div class="c b">ホームセンター ミドリ</div>
<div class="c">世田谷店 TEL 03-5432-1180</div>
<div class="c">2026年11月18日(水) 14:07</div><hr>
<div class="row"><span>梱包用テープ 透明 x6</span><span>¥2,340</span></div>
<div class="row"><span>エアキャップ 1200mm</span><span>¥3,980</span></div>
<div class="row"><span>ダンボール 80サイズ x20</span><span>¥4,400</span></div>
<div class="row"><span>カッターナイフ 替刃</span><span>¥580</span></div><hr>
<div class="row"><span>小計</span><span>¥11,300</span></div>
<div class="row"><span>消費税 10%</span><span>¥1,130</span></div>
<div class="row t"><span>合計</span><span>¥12,430</span></div><hr>
<div class="row"><span>お預り(クレジット)</span><span>¥12,430</span></div>
<div class="c">登録番号 T7010901045821</div>
<div class="c">ありがとうございました</div>
</div></body></html>"""

ERROR_SCREEN = """
<html><head><style>""" + BASE_CSS + """
body { background: #eef1f5; width: 1280px; height: 800px; }
.top { height: 56px; background: #1f3b5a; color: #fff; display: flex; align-items: center; padding: 0 24px; font-weight: 700; font-size: 18px; gap: 28px; }
.top span { font-weight: 400; opacity: .8; font-size: 15px; }
.main { padding: 28px 40px; } h2 { margin: 0 0 16px; }
table { width: 100%; background: #fff; border-collapse: collapse; font-size: 14px; }
td, th { padding: 10px 12px; border-bottom: 1px solid #e3e6ea; text-align: left; }
.mask { position: fixed; inset: 0; background: rgba(0,0,0,.35); }
.dlg { position: fixed; top: 210px; left: 390px; width: 500px; background: #fff; border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,.3); padding: 26px 28px; }
.dlg h3 { margin: 0 0 10px; color: #b42318; font-size: 20px; }
.dlg p { margin: 6px 0; font-size: 15px; } code { background: #f4f4f4; padding: 2px 6px; }
.btns { margin-top: 20px; text-align: right; } .btns button { padding: 8px 18px; border-radius: 6px; border: 1px solid #ccc; background: #fff; margin-left: 8px; font-size: 14px; }
.btns .p { background: #1f3b5a; color: #fff; border: none; }
</style></head><body>
<div class="top">Hikari Express Business Portal <span>Shipments</span><span>Labels</span><span>Invoices</span><span style="margin-left:auto">Kumo Supply K.K.</span></div>
<div class="main"><h2>Batch label creation (Nov 21, 2026)</h2>
<table><tr><th>Batch</th><th>Orders</th><th>Status</th><th>Created</th></tr>
<tr><td>B-20261121-03</td><td>23</td><td>Processing…</td><td>09:12</td></tr>
<tr><td>B-20261121-02</td><td>41</td><td>Completed</td><td>08:47</td></tr>
<tr><td>B-20261121-01</td><td>18</td><td>Completed</td><td>08:02</td></tr></table></div>
<div class="mask"></div>
<div class="dlg"><h3>⚠ Label generation failed</h3>
<p>Batch <b>B-20261121-03</b>: 0 of 23 labels created.</p>
<p>Error <code>ERR_ADDR_SCHEMA_V3</code>: the field <code>postal_code</code> must match the pattern <code>NNN-NNNN</code>.</p>
<p>No labels were created for this batch. Orders remain in "Ready to ship".</p>
<div class="btns"><button>Download error log</button><button class="p">Close</button></div></div>
</body></html>"""

SHIP_LABEL = """
<html><head><style>""" + BASE_CSS + """
body { background: transparent; }
.l { width: 560px; background: #fff; border: 3px solid #111; font-size: 17px; }
.h { background: #111; color: #fff; padding: 8px 14px; font-weight: 700; font-size: 20px; display: flex; justify-content: space-between; }
.s { padding: 10px 14px; border-bottom: 2px solid #111; } .k { font-size: 12px; color: #444; }
.big { font-size: 24px; font-weight: 700; letter-spacing: 1px; } .zip { font-size: 30px; font-weight: 800; }
.bar { height: 70px; margin: 10px 14px; background: repeating-linear-gradient(90deg,#111 0 3px,#fff 3px 5px,#111 5px 6px,#fff 6px 9px,#111 9px 13px,#fff 13px 15px); }
</style></head><body><div class="l" id="cap">
<div class="h"><span>ヒカリ急便 HIKARI EXPRESS</span><span>宅配便</span></div>
<div class="s"><div class="k">お届け先 / TO</div>
<div class="zip">〒 <span id="z">231-0023</span></div>
<div class="big">神奈川県横浜市中区山下町 27-4 ベイサイド303</div>
<div class="big">佐藤 健一 様</div><div>TEL 045-662-3091</div></div>
<div class="s"><div class="k">ご依頼主 / FROM</div><div>クモサプライ株式会社 東京都品川区大崎 2-1-9</div><div>TEL 03-6420-5517</div></div>
<div class="s" style="display:flex;justify-content:space-between"><span>注文 KS-10455</span><span>品名: 食器 (割れ物注意)</span><span>1/1</span></div>
<div class="bar"></div><div style="text-align:center;padding-bottom:8px">4427 1180 9036</div>
</div></body></html>"""

# ------------------------------------------------------------------ PDFs ---
INVOICE = """
<html><head><style>""" + BASE_CSS + """
@page { size: A4; margin: 18mm; } body { font-size: 12px; }
h1 { font-size: 22px; margin: 0; } .muted { color: #555; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; } th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: right; } th:first-child, td:first-child { text-align: left; }
.hdr { display: flex; justify-content: space-between; } .box { border: 1px solid #333; padding: 10px; margin-top: 14px; }
.tot td { font-weight: 700; } .pb { page-break-before: always; }
</style></head><body>
<div class="hdr"><div><h1>INVOICE</h1><div class="muted">Hikari Express Co., Ltd. · 3-8-1 Shiba, Minato-ku, Tokyo</div></div>
<div style="text-align:right">Invoice no. HX-2026-11-08812<br>Issue date: 2026-11-20<br>Due date: 2026-12-10</div></div>
<div class="box"><b>Bill to:</b> Kumo Supply K.K., 2-1-9 Osaki, Shinagawa-ku, Tokyo<br>Customer ID: KMS-0417 · Contract: HX-CORP-2025-117 (contract fuel surcharge rate: <b>12%</b>)</div>
<table><tr><th>Description</th><th>Quantity</th><th>Unit price (¥)</th><th>Amount (¥)</th></tr>
<tr><td>Parcel delivery, size 60–80 (Nov 1–15, 2026)</td><td>412</td><td>820</td><td>337,840</td></tr>
<tr><td>Fuel surcharge (18%)</td><td></td><td></td><td>60,811</td></tr>
<tr><td>Redelivery fee</td><td>9</td><td>330</td><td>2,970</td></tr>
<tr><td>Subtotal</td><td></td><td></td><td>401,621</td></tr>
<tr><td>Consumption tax (10%)</td><td></td><td></td><td>40,162</td></tr>
<tr class="tot"><td>Total due</td><td></td><td></td><td>441,783</td></tr></table>
<p class="muted">Please transfer the total due to MUFG Bank, Shiba branch, ordinary 0482913, Hikari Express Co., Ltd.</p>
<div class="pb"></div>
<h2>Annex A: Daily parcel count (Nov 1–15, 2026)</h2>
<table><tr><th>Date</th><th>Parcels</th><th>Date</th><th>Parcels</th></tr>
<tr><td>Nov 1</td><td>22</td><td>Nov 9</td><td>19</td></tr>
<tr><td>Nov 2</td><td>17</td><td>Nov 10</td><td>31</td></tr>
<tr><td>Nov 3</td><td>28</td><td>Nov 11</td><td>29</td></tr>
<tr><td>Nov 4</td><td>30</td><td>Nov 12</td><td>27</td></tr>
<tr><td>Nov 5</td><td>26</td><td>Nov 13</td><td>25</td></tr>
<tr><td>Nov 6</td><td>24</td><td>Nov 14</td><td>32</td></tr>
<tr><td>Nov 7</td><td>27</td><td>Nov 15</td><td>21</td></tr>
<tr><td>Nov 8</td><td>40</td><td></td><td></td></tr>
<tr class="tot"><td>Total</td><td colspan="3">398 parcels</td></tr></table>
</body></html>"""

CONTRACT = """
<html><head><style>""" + BASE_CSS + """
@page { size: A4; margin: 22mm; } body { font-family: 'Noto Serif CJK TC', Georgia, serif; font-size: 12.5px; line-height: 1.6; }
h1 { text-align: center; font-size: 20px; } h3 { font-size: 13.5px; margin: 16px 0 4px; } p { margin: 4px 0; text-align: justify; }
</style></head><body>
<h1>SOFTWARE SUBSCRIPTION AGREEMENT</h1>
<p style="text-align:center">between ShelfWise Inc. ("Provider") and Kumo Supply K.K. ("Customer")<br>Order form SW-KMS-2026-03 · Effective date: December 1, 2026</p>
""" + "".join(f"""<h3>{n}. {t}</h3><p>{b}</p>""" for n, t, b in [
    (1, "Definitions", "\"Service\" means the ShelfWise warehouse management software made available by Provider as a hosted service, including the inventory, picking and label modules listed in the Order Form. \"Users\" means employees and contractors of Customer authorized to use the Service. \"Customer Data\" means all data submitted by Customer to the Service."),
    (2, "Subscription", "Provider grants Customer a non-exclusive, non-transferable right to access and use the Service during the Subscription Term for Customer's internal business purposes, for up to 15 named Users and 2 warehouse locations."),
    (3, "Fees and payment", "Customer shall pay the annual subscription fee of JPY 1,560,000 (excluding consumption tax) in advance, within thirty (30) days of the invoice date. Late payments bear interest of 1% per month. Fees are non-refundable except as expressly provided in this Agreement."),
    (4, "Customer obligations", "Customer is responsible for the accuracy of Customer Data, for maintaining the confidentiality of User credentials, and for compliance with applicable laws in its use of the Service."),
    (5, "Service levels", "Provider will use commercially reasonable efforts to make the Service available 99.5% of the time in each calendar month, excluding scheduled maintenance announced at least 48 hours in advance. Service credits are described in Schedule B."),
    (6, "Data protection", "Provider processes Customer Data only to provide the Service and in accordance with the Data Processing Addendum. Customer Data is stored in data centers located in Japan."),
    (7, "Confidentiality", "Each party shall keep the other party's Confidential Information confidential and use it only for the purposes of this Agreement, during the term and for three (3) years thereafter."),
    (8, "Warranties and liability", "Except as expressly stated, the Service is provided \"as is\". Each party's aggregate liability under this Agreement is limited to the fees paid by Customer in the twelve (12) months preceding the claim."),
    (9, "Term and renewal", "9.1 The initial Subscription Term is twelve (12) months from the Effective Date. 9.2 Thereafter, this Agreement shall automatically renew for successive renewal terms of twenty-four (24) months each, unless either party gives written notice of non-renewal at least ninety (90) days before the end of the then-current term. 9.3 Provider may increase the fees for any renewal term by up to fifteen percent (15%) by notice given at least sixty (60) days before the renewal."),
    (10, "Termination", "Either party may terminate this Agreement for material breach that remains uncured thirty (30) days after written notice. On termination, Customer may export Customer Data for thirty (30) days."),
    (11, "Governing law", "This Agreement is governed by the laws of Japan. The Tokyo District Court has exclusive jurisdiction of first instance."),
    (12, "General", "This Agreement, including its Order Form and Schedules, is the entire agreement between the parties regarding its subject matter. Amendments must be in writing and signed by both parties."),
]) + """
<p style="margin-top:28px">Signed for ShelfWise Inc.: ____________________ &nbsp;&nbsp; Signed for Kumo Supply K.K.: ____________________</p>
</body></html>"""

DELIVERY_NOTE = """
<html><head><style>""" + BASE_CSS + """
body { width: 1240px; height: 1754px; background: #fff; padding: 90px 100px; font-family: 'Noto Serif CJK TC', serif; font-size: 26px; }
h1 { text-align: center; letter-spacing: 20px; font-size: 48px; border-bottom: 4px double #000; padding-bottom: 10px; }
table { width: 100%; border-collapse: collapse; margin-top: 30px; } th, td { border: 2px solid #000; padding: 14px; } td.n { text-align: right; }
.stamp { display: inline-block; border: 4px solid #c00; color: #c00; border-radius: 50%; width: 120px; height: 120px; text-align: center; line-height: 112px; font-size: 30px; transform: rotate(-12deg); }
</style></head><body>
<h1>納品書</h1>
<div style="display:flex;justify-content:space-between;margin-top:30px">
<div>クモサプライ株式会社 御中<br>東京都品川区大崎2-1-9<br><br>下記の通り納品いたします。</div>
<div style="text-align:right">納品日: 2026年11月19日<br>伝票番号: DN-58821<br>大和梱包工業株式会社<br>埼玉県川口市領家5-14-2<br><span class="stamp">大和</span></div></div>
<table><tr><th>品名</th><th>発注数</th><th>納品数</th><th>備考</th></tr>
<tr><td>ダンボール 80サイズ (A式)</td><td class="n">1,000</td><td class="n">1,000</td><td></td></tr>
<tr><td>ダンボール 100サイズ (A式)</td><td class="n">600</td><td class="n">400</td><td>欠品 200 (12月2日入荷予定)</td></tr>
<tr><td>緩衝材 (紙製) 10kg</td><td class="n">400</td><td class="n">400</td><td></td></tr>
<tr><td><b>合計</b></td><td class="n"><b>2,000</b></td><td class="n"><b>1,800</b></td><td></td></tr></table>
<p style="margin-top:40px">受領印: ____________ &nbsp;&nbsp;&nbsp; 備考: 欠品分は入荷次第、別便にてお届けいたします。</p>
</body></html>"""

CLAIM = """
<html><head><style>""" + BASE_CSS + """
@page { size: A4; margin: 18mm; } body { font-size: 13px; }
h1 { font-size: 21px; margin-bottom: 2px; } .f { border: 1px solid #999; margin: 10px 0; }
.f div { display: flex; border-bottom: 1px solid #ddd; } .f div:last-child { border-bottom: none; }
.f b { width: 220px; background: #f3f4f6; padding: 8px 10px; font-weight: 600; } .f span { padding: 8px 10px; flex: 1; font-family: 'Comic Relief', cursive; font-size: 15px; color: #1a3d8f; }
</style></head><body>
<h1>Kumo Supply: Damaged or Missing Item Claim</h1><div>Please fill in this form and email it to support@kumo-supply.example within 14 days of delivery.</div>
<div class="f">
<div><b>Full name</b><span>Emily Tanaka-Hughes</span></div>
<div><b>Email</b><span>emily.th@example.com</span></div>
<div><b>Phone</b><span>080-2295-4713</span></div>
<div><b>Delivery address</b><span>4-12-6 Kichijoji Honcho, Musashino-shi, Tokyo 180-0004</span></div>
<div><b>Order number</b><span>KS-10377</span></div>
<div><b>Delivery date</b><span>November 18, 2026</span></div>
<div><b>Carrier</b><span>Hikari Express</span></div>
<div><b>What was damaged or missing?</b><span>The outer box was crushed on one corner. 3 of the 12 stoneware mugs (item MG-STN-12, ¥1,800 each) arrived broken. The other 9 are fine.</span></div>
<div><b>Amount claimed</b><span>¥5,400 (3 mugs × ¥1,800)</span></div>
<div><b>Preferred resolution</b><span>☑ Refund to original payment method &nbsp; ☐ Replacement</span></div>
<div><b>Photos</b><span>Attached separately (4 photos of box and mugs)</span></div>
<div><b>Signature / date</b><span>E. Tanaka-Hughes, 2026-11-19</span></div>
</div><p style="color:#666">For office use: claim no. ______ · approved by ______</p>
</body></html>"""

# ---------------------------------------------------------------- helpers ---
def render_png(pg, html, path, selector=None, size=(1280, 800)):
    pg.set_viewport_size({"width": size[0], "height": size[1]})
    pg.set_content(html); pg.wait_for_timeout(300)
    (pg.locator(selector) if selector else pg).screenshot(path=path, omit_background=bool(selector))

def render_pdf(pg, html, path):
    pg.set_content(html); pg.wait_for_timeout(300)
    pg.pdf(path=path, format="A4", print_background=True)

def texture(h, w, base, var, grain):
    noise = rng.normal(0, var, (h // grain + 1, w // grain + 1, 1))
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    return np.clip(np.array(base, np.float32)[None, None, :] + noise + rng.normal(0, 4, (h, w, 1)), 0, 255)

def photo_of(src_png, out_jpg, bg_base, angle_pts, blur, shade, size=(1600, 1200), smudge=None):
    """Paste a rendered item onto a textured surface, warp it like a phone photo, add light, blur and JPEG noise."""
    item = cv2.imread(src_png, cv2.IMREAD_UNCHANGED)
    if item.shape[2] == 3: item = np.dstack([item, np.full(item.shape[:2], 255, np.uint8)])
    if smudge:
        x, y, r = smudge
        m = np.zeros(item.shape[:2], np.float32)
        cv2.ellipse(m, (x, y), (r, int(r * .55)), -8, 0, 360, 1.0, -1)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), r / 6) * 1.5, 0, 0.97)[..., None]  # a dark water stain over the postal code
        stain = np.array([80, 72, 68, 255], np.float32)
        item = (item.astype(np.float32) * (1 - m) + stain * m).astype(np.uint8)
    h, w = item.shape[:2]; W, H = size
    bg = texture(H, W, bg_base, 18, 40)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]); dst = np.float32(angle_pts)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(item, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255
    shadow = cv2.GaussianBlur(a, (0, 0), 14)[..., None] if a.ndim == 2 else cv2.GaussianBlur(a, (0, 0), 14)
    shadow = shadow.reshape(H, W, 1)
    bg = bg * (1 - 0.35 * np.roll(np.roll(shadow, 14, 0), 10, 1))
    img = bg * (1 - a) + warped[..., :3].astype(np.float32) * a
    yy, xx = np.mgrid[0:H, 0:W]
    light = 1 - shade * ((xx / W) * 0.6 + (yy / H) * 0.4)
    img = np.clip(img * light[..., None] + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    if blur: img = cv2.GaussianBlur(img, (0, 0), blur)
    cv2.imwrite(out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 72])

def scan_pdf(src_png, out_pdf):
    """Turn a rendered page into an image-only 'scanned' PDF: grey, rotated, noisy, no text layer."""
    im = cv2.imread(src_png, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    h, w = im.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.3, 1.0)
    im = cv2.warpAffine(im, M, (w, h), borderValue=235)
    im = im * 0.9 + 18 + rng.normal(0, 9, im.shape)
    im[:, :25] *= 0.6  # dark scanner edge
    im = cv2.GaussianBlur(np.clip(im, 0, 255), (0, 0), 0.8).astype(np.uint8)
    Image.fromarray(im).convert("L").save(out_pdf, "PDF", resolution=150)

# ---------------------------------------------------------------- videos ---
APP = """
<html><head><style>""" + BASE_CSS + """
body { background: #f4f6f9; width: 1280px; height: 720px; overflow: hidden; }
.nav { height: 52px; background: #0f5c4d; color: #fff; display: flex; align-items: center; gap: 26px; padding: 0 24px; font-weight: 700; }
.nav span { font-weight: 400; opacity: .85; } .nav .on { opacity: 1; border-bottom: 2px solid #fff; padding-bottom: 4px; }
.page { padding: 26px 36px; } h2 { margin: 0 0 14px; }
.card { background: #fff; border-radius: 8px; padding: 18px 22px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 14px; }
.btn { display: inline-block; background: #0f5c4d; color: #fff; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: 600; }
.toast { position: fixed; top: 70px; right: 30px; background: #b42318; color: #fff; padding: 14px 18px; border-radius: 8px; display: none; font-size: 15px; box-shadow: 0 6px 20px rgba(0,0,0,.25); }
table { width: 100%; border-collapse: collapse; font-size: 14px; } td, th { padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }
.red { color: #b42318; font-weight: 700; } .muted { color: #666; } input { padding: 8px; width: 260px; border: 1px solid #ccc; border-radius: 6px; }
#cursor { position: fixed; width: 18px; height: 18px; border-radius: 50%; background: rgba(0,0,0,.35); border: 2px solid #fff; pointer-events: none; z-index: 99; transition: all .6s ease; left: 600px; top: 400px; }
</style></head><body><div id="cursor"></div><div id="root"></div>
<script>
const nav = on => `<div class="nav">Kumo Admin ${['Orders','Shipping','Tracking','Products','Payments'].map(x=>`<span class="${x==on?'on':''}">${x}</span>`).join('')}<span style="margin-left:auto">ops@kumo-supply</span></div>`;
const views = {
 label: () => nav('Shipping') + `<div class="page"><h2>Create shipping label</h2>
  <div class="card"><b>Order KS-10466</b> · 2 items · Recipient: Yamada T., 〒150-0001 Tokyo Shibuya-ku Jingumae 4-3-15<br><span class="muted">Carrier: Hikari Express · Size 80</span></div>
  <div class="card"><span class="btn" id="go">Create label</span> <span class="muted" id="st" style="margin-left:14px">Status: Ready to ship</span></div></div>
  <div class="toast" id="t">Address validation failed (ERR_ADDR_SCHEMA_V3). Label not created.</div>`,
 checkout: () => `<div class="nav">Kumo Supply · Checkout</div><div class="page"><h2>Payment</h2>
  <div class="card">Order KS-10391 · Kraft boxes size M × 40 · <b>¥8,640</b></div>
  <div class="card">Card: VISA •••• 4471 · Haruka Mori<br><br><span class="btn" id="go">Pay ¥8,640</span> <span id="st" class="muted" style="margin-left:14px"></span></div>
  <div class="card" id="hist" style="display:none"><b>Order history</b><table><tr><th>Date</th><th>Order</th><th>Amount</th><th>Status</th></tr>
  <tr><td>Nov 14</td><td>KS-10391</td><td>¥8,640</td><td>Paid</td></tr><tr><td>Nov 15</td><td>KS-10391</td><td>¥8,640</td><td class="red">Paid (duplicate)</td></tr></table></div></div>`,
 tracking: () => nav('Tracking') + `<div class="page"><h2>Shipments in transit</h2>
  <div class="card">Filter: <input value="carrier: Hikari Express · status: delayed"> <span class="muted" style="margin-left:12px">Updated 11:40</span></div>
  <div class="card"><div class="red" style="margin-bottom:8px">37 shipments delayed at Hikari Express Saitama hub (since Nov 20)</div>
  <table><tr><th>Order</th><th>Destination</th><th>Last scan</th><th>Status</th></tr>` +
  Array.from({length:8},(_, i)=>`<tr><td>KS-10${420+i*3}</td><td>${['Osaka','Nagoya','Sendai','Fukuoka','Kobe','Sapporo','Kyoto','Hiroshima'][i]}</td><td>Nov 20 Saitama hub</td><td class="red">Delayed</td></tr>`).join('') +
  `<tr><td colspan="4" class="muted">… 29 more</td></tr></table></div></div>`,
 product: () => nav('Products') + `<div class="page"><h2>New product: Eco Mailer Box (preview)</h2>
  <div class="card" style="display:flex;gap:26px"><div style="width:300px;height:220px;background:linear-gradient(135deg,#c8a27a,#a47b52);border-radius:8px"></div>
  <div><h3 style="margin-top:0">Eco Mailer Box · 100% recycled kraft</h3><p>Sizes S / M / L · from ¥58 per box</p><p class="muted">Plastic-free tape strip, prints in 1–2 colors.</p><span class="btn">Add to catalog</span></div></div>
  <div class="card muted">Launch planned for January 2027. This is a walkthrough for the team.</div></div>`
};
function show(v){ document.getElementById('root').innerHTML = views[v](); }
function move(x,y){ const c=document.getElementById('cursor'); c.style.left=x+'px'; c.style.top=y+'px'; }
</script></body></html>"""

def record_videos(browser, outdir, tmp):
    scenes = {
        "V1_label_error_flash": [("show", "label"), ("wait", 3000), ("move", (90, 262)), ("wait", 1500),
            ("js", "document.getElementById('t').style.display='block';document.getElementById('st').textContent='Status: Ready to ship'"),
            ("wait", 2000), ("js", "document.getElementById('t').style.display='none'"), ("move", (700, 450)), ("wait", 9000)],
        "V2_checkout_double_charge": [("show", "checkout"), ("wait", 3000), ("move", (110, 300)), ("wait", 1500),
            ("js", "document.getElementById('st').textContent='Payment failed. Please try again.'"), ("wait", 3000),
            ("js", "document.getElementById('st').textContent='Processing…'"), ("wait", 2000),
            ("js", "document.getElementById('st').textContent='Payment complete';document.getElementById('hist').style.display='block'"),
            ("move", (420, 520)), ("wait", 7000)],
        "V3_tracking_delays": [("show", "tracking"), ("wait", 3000), ("move", (300, 200)), ("wait", 4000),
            ("move", (500, 420)), ("wait", 4000), ("js", "window.scrollTo(0,200)"), ("wait", 5000)],
        "V4_product_preview": [("show", "product"), ("wait", 3000), ("move", (200, 250)), ("wait", 4000),
            ("move", (620, 300)), ("wait", 4000), ("move", (680, 360)), ("wait", 5000)],
    }
    for name, steps in scenes.items():
        ctx = browser.new_context(viewport={"width": 1280, "height": 720},
                                  record_video_dir=tmp, record_video_size={"width": 1280, "height": 720})
        pg = ctx.new_page(); pg.set_content(APP)
        for kind, arg in steps:
            if kind == "show": pg.evaluate(f"show('{arg}')")
            elif kind == "wait": pg.wait_for_timeout(arg)
            elif kind == "move": pg.evaluate(f"move({arg[0]},{arg[1]})")
            elif kind == "js": pg.evaluate(arg)
        webm = pg.video.path(); ctx.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-pix_fmt", "yuv420p", "-an", os.path.join(outdir, name + ".mp4")], check=True)

def main():
    for d in ("image", "pdf", "video"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    tmp = tempfile.mkdtemp()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = b.new_page()
        hi = b.new_page(device_scale_factor=2)  # sharper source for the photo items
        # images
        render_png(hi, RECEIPT, f"{tmp}/receipt.png", "#cap")
        photo_of(f"{tmp}/receipt.png", f"{OUT}/image/I1_receipt_photo.jpg", (120, 140, 160),
                 [[560, 170], [1010, 230], [930, 1130], [430, 1060]], blur=1.6, shade=0.45)
        render_png(pg, ERROR_SCREEN, f"{OUT}/image/I2_carrier_portal_error.png")
        render_png(hi, SHIP_LABEL, f"{tmp}/label.png", "#cap")
        photo_of(f"{tmp}/label.png", f"{OUT}/image/I4_shipping_label_photo.jpg", (150, 175, 200),
                 [[330, 260], [1260, 330], [1210, 960], [290, 900]], blur=1.1, shade=0.3, smudge=(295, 165, 130))
        # PDFs
        render_pdf(pg, INVOICE, f"{OUT}/pdf/P1_carrier_invoice.pdf")
        render_pdf(pg, CONTRACT, f"{OUT}/pdf/P2_software_contract.pdf")
        render_png(pg, DELIVERY_NOTE, f"{tmp}/dn.png", size=(1240, 1754))
        scan_pdf(f"{tmp}/dn.png", f"{OUT}/pdf/P3_scanned_delivery_note_ja.pdf")
        render_pdf(pg, CLAIM, f"{OUT}/pdf/P4_damage_claim_form.pdf")
        # videos
        record_videos(b, f"{OUT}/video", tmp)
        b.close()
    shutil.rmtree(tmp)
    print("built into", OUT)

if __name__ == "__main__":
    main()
