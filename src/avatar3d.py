"""3D Avatar Sign Language Generator module for SignBridge.

Renders an appealing, authentic, friendly human 3D avatar with natural skin tones,
warm portrait studio lighting, expressive facial features, lifelike eyes with catchlights,
fitted modern clothing, tri-phalangeal finger kinematics, and distinct, linguistically accurate
sign language vocabulary for all 20 signs.
"""

from __future__ import annotations

import json
import streamlit.components.v1 as components


# The complete list of all 20 signs in the system
ALL_20_SIGNS = [
    ("hello", "Hello"),
    ("thank_you", "Thank You"),
    ("please", "Please"),
    ("sorry", "Sorry"),
    ("yes", "Yes"),
    ("no", "No"),
    ("help", "Help"),
    ("water", "Water"),
    ("food", "Food"),
    ("medicine", "Medicine"),
    ("doctor", "Doctor"),
    ("family", "Family"),
    ("work", "Work"),
    ("school", "School"),
    ("home", "Home"),
    ("money", "Money"),
    ("phone", "Phone"),
    ("happy", "Happy"),
    ("sad", "Sad"),
    ("pain", "Pain"),
]


def get_3d_avatar_html(initial_sign: str = "hello") -> str:
    """Generate self-contained HTML/WebGL Three.js canvas for realistic 3D human sign language avatar."""
    safe_sign = json.dumps(str(initial_sign).strip().lower().replace(" ", "_") if initial_sign else "hello")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Sign Language Human Avatar</title>
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    user-select: none;
  }}
  body {{
    background: #0B0F19;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden;
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  #avatar-container {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: radial-gradient(circle at 50% 30%, #1e293b 0%, #090d16 100%);
  }}
  #canvas3d {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  /* Top HUD overlay */
  .hud-top {{
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 8px 14px;
    z-index: 10;
  }}
  .badge-active {{
    background: linear-gradient(135deg, #10B981, #06B6D4);
    color: #FFFFFF;
    padding: 4px 14px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: all 0.25s ease;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
  }}
  .hud-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  /* Controls deck at bottom */
  .hud-bottom {{
    position: absolute;
    bottom: 10px;
    left: 10px;
    right: 10px;
    background: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 10px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 10;
  }}
  .speed-select {{
    background: rgba(51, 65, 85, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #E2E8F0;
    font-size: 0.82rem;
    padding: 5px 10px;
    border-radius: 8px;
    cursor: pointer;
    outline: none;
  }}

  /* Chips Grid */
  .chips-container {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .chips-label {{
    font-size: 0.76rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .chips-grid {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 2px;
  }}
  .chips-grid::-webkit-scrollbar {{
    height: 4px;
  }}
  .chips-grid::-webkit-scrollbar-thumb {{
    background: rgba(255, 255, 255, 0.25);
    border-radius: 4px;
  }}
  .chip {{
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #CBD5E1;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 5px 12px;
    border-radius: 12px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
  }}
  .chip:hover {{
    background: #4F46E5;
    color: #FFFFFF;
    border-color: #818CF8;
    transform: translateY(-1px);
  }}
  .chip.active {{
    background: linear-gradient(135deg, #4F46E5, #06B6D4);
    color: #FFFFFF;
    border-color: #38BDF8;
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.45);
    font-weight: 700;
  }}

  .orbit-hint {{
    position: absolute;
    top: 55px;
    right: 12px;
    font-size: 0.72rem;
    color: #64748B;
    background: rgba(15, 23, 42, 0.65);
    padding: 3px 8px;
    border-radius: 6px;
    pointer-events: none;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

<div id="avatar-container">
  <!-- Top HUD -->
  <div class="hud-top">
    <div class="hud-title">
      <span>👤 3D Human Signer</span>
      <span style="color:#64748B; font-weight:400; font-size:0.8rem;">| Natural Human Model</span>
    </div>
    <div id="statusBadge" class="badge-active">Ready</div>
  </div>

  <div class="orbit-hint">🖱️ Drag to rotate | Scroll to zoom</div>

  <canvas id="canvas3d"></canvas>

  <!-- Bottom Controls Deck -->
  <div class="hud-bottom">
    <div class="chips-container">
      <div class="chips-label">
        <span style="font-weight:600; color:#E2E8F0;">Select Sign:</span>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="color:#64748B; font-size:0.75rem;">Speed:</span>
          <select id="speedSelect" class="speed-select" title="Animation Speed">
            <option value="0.7">0.7x (Slow)</option>
            <option value="1.0" selected>1.0x (Normal)</option>
            <option value="1.3">1.3x (Fast)</option>
          </select>
        </div>
      </div>
      <div class="chips-grid" id="chipsGrid">
        <span class="chip" data-sign="hello" onclick="selectSign('hello')">Hello</span>
        <span class="chip" data-sign="thank_you" onclick="selectSign('thank_you')">Thank You</span>
        <span class="chip" data-sign="please" onclick="selectSign('please')">Please</span>
        <span class="chip" data-sign="sorry" onclick="selectSign('sorry')">Sorry</span>
        <span class="chip" data-sign="yes" onclick="selectSign('yes')">Yes</span>
        <span class="chip" data-sign="no" onclick="selectSign('no')">No</span>
        <span class="chip" data-sign="help" onclick="selectSign('help')">Help</span>
        <span class="chip" data-sign="water" onclick="selectSign('water')">Water</span>
        <span class="chip" data-sign="food" onclick="selectSign('food')">Food</span>
        <span class="chip" data-sign="medicine" onclick="selectSign('medicine')">Medicine</span>
        <span class="chip" data-sign="doctor" onclick="selectSign('doctor')">Doctor</span>
        <span class="chip" data-sign="family" onclick="selectSign('family')">Family</span>
        <span class="chip" data-sign="work" onclick="selectSign('work')">Work</span>
        <span class="chip" data-sign="school" onclick="selectSign('school')">School</span>
        <span class="chip" data-sign="home" onclick="selectSign('home')">Home</span>
        <span class="chip" data-sign="money" onclick="selectSign('money')">Money</span>
        <span class="chip" data-sign="phone" onclick="selectSign('phone')">Phone</span>
        <span class="chip" data-sign="happy" onclick="selectSign('happy')">Happy</span>
        <span class="chip" data-sign="sad" onclick="selectSign('sad')">Sad</span>
        <span class="chip" data-sign="pain" onclick="selectSign('pain')">Pain</span>
      </div>
    </div>
  </div>
</div>

<script>
// ============================================================================
// 1. PROCEDURAL ORGANIC TEXTURES (WARM NATURAL SKIN, FABRIC, IRIS, HAIR)
// ============================================================================
function createSkinTexture() {{
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext('2d');

  // Warm natural human skin base
  ctx.fillStyle = '#f0c8ab';
  ctx.fillRect(0, 0, 512, 512);

  // Soft peach/rosy subcutaneous flush
  const grad = ctx.createRadialGradient(256, 256, 30, 256, 256, 250);
  grad.addColorStop(0, 'rgba(235, 120, 100, 0.14)');
  grad.addColorStop(0.6, 'rgba(230, 135, 110, 0.06)');
  grad.addColorStop(1, 'rgba(240, 200, 170, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);

  // Subtle natural skin micro-grain
  for (let i = 0; i < 20000; i++) {{
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const rad = Math.random() * 1.5 + 0.5;
    ctx.fillStyle = Math.random() > 0.5 ? 'rgba(190, 115, 90, 0.035)' : 'rgba(255, 235, 215, 0.045)';
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fill();
  }}

  return new THREE.CanvasTexture(c);
}}

function createClothTexture() {{
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#1e3a8a';
  ctx.fillRect(0, 0, 256, 256);

  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 1.0;
  for (let i = 0; i < 256; i += 4) {{
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 256);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(256, i);
    ctx.stroke();
  }}
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 4);
  return tex;
}}

function createIrisTexture() {{
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 256;
  const ctx = c.getContext('2d');

  const grad = ctx.createRadialGradient(128, 128, 20, 128, 128, 120);
  grad.addColorStop(0, '#1c130d');
  grad.addColorStop(0.3, '#3d2516');
  grad.addColorStop(0.7, '#633d24');
  grad.addColorStop(0.9, '#4a2c19');
  grad.addColorStop(1.0, '#150d09');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);

  ctx.strokeStyle = 'rgba(215, 160, 100, 0.22)';
  for (let a = 0; a < Math.PI * 2; a += 0.06) {{
    ctx.beginPath();
    ctx.moveTo(128 + Math.cos(a) * 35, 128 + Math.sin(a) * 35);
    ctx.lineTo(128 + Math.cos(a) * 115, 128 + Math.sin(a) * 115);
    ctx.stroke();
  }}

  ctx.strokeStyle = '#0d0805';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(128, 128, 118, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#050302';
  ctx.beginPath();
  ctx.arc(128, 128, 42, 0, Math.PI * 2);
  ctx.fill();

  return new THREE.CanvasTexture(c);
}}

const skinTex = createSkinTexture();
const clothTex = createClothTexture();
const irisTex = createIrisTexture();

// ============================================================================
// 2. SCENE, CAMERA & WARM NATURAL PORTRAIT STUDIO LIGHTING
// ============================================================================
const container = document.getElementById('avatar-container');
const canvas = document.getElementById('canvas3d');

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(34, container.clientWidth / container.clientHeight, 0.1, 50);
camera.position.set(0, 1.30, 2.30);

const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: false }});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.15, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 1.2;
controls.maxDistance = 3.4;
controls.maxPolarAngle = Math.PI / 2 + 0.1;
controls.update();

// Warm, flattering portrait lighting
const ambientLight = new THREE.AmbientLight(0xfff8f2, 0.92);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xfff2e0, 1.25);
keyLight.position.set(1.4, 2.8, 2.2);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffeedd, 0.75);
fillLight.position.set(-1.8, 1.8, 1.8);
scene.add(fillLight);

const backLight = new THREE.DirectionalLight(0xffecd6, 0.65);
backLight.position.set(0, 2.2, -2.2);
scene.add(backLight);

const shadowGeo = new THREE.CircleGeometry(0.55, 36);
const shadowMat = new THREE.MeshBasicMaterial({{ color: 0x050811, transparent: true, opacity: 0.45 }});
const groundShadow = new THREE.Mesh(shadowGeo, shadowMat);
groundShadow.rotation.x = -Math.PI / 2;
groundShadow.position.y = 0.05;
scene.add(groundShadow);

// ============================================================================
// 3. CLEAN, ATTRACTIVE HUMAN ANATOMY & SCULPTED FEATURES
// ============================================================================
const skinMat = new THREE.MeshStandardMaterial({{
  map: skinTex,
  color: 0xffdcc4,
  roughness: 0.58,
  metalness: 0.02,
}});

const blushMat = new THREE.MeshStandardMaterial({{
  color: 0xdf8276,
  roughness: 0.62,
  metalness: 0.01,
  transparent: true,
  opacity: 0.55,
}});

const lipsMat = new THREE.MeshStandardMaterial({{
  color: 0xc46960,
  roughness: 0.42,
  metalness: 0.04,
}});

const shirtMat = new THREE.MeshStandardMaterial({{
  map: clothTex,
  color: 0x1e3a8a,
  roughness: 0.70,
  metalness: 0.08,
}});

const collarMat = new THREE.MeshStandardMaterial({{
  color: 0x2563eb,
  roughness: 0.75,
  metalness: 0.05,
}});

const hairMat = new THREE.MeshStandardMaterial({{
  color: 0x241b17,
  roughness: 0.55,
  metalness: 0.10,
}});

const eyeScleraMat = new THREE.MeshStandardMaterial({{
  color: 0xffffff,
  roughness: 0.10,
  metalness: 0.02,
}});

const irisMat = new THREE.MeshStandardMaterial({{
  map: irisTex,
  roughness: 0.10,
  metalness: 0.05,
}});

const catchlightMat = new THREE.MeshBasicMaterial({{
  color: 0xffffff,
}});

const browMat = new THREE.MeshStandardMaterial({{
  color: 0x2b1e19,
  roughness: 0.75,
  metalness: 0.02,
}});

const nailMat = new THREE.MeshStandardMaterial({{
  color: 0xfce4d6,
  roughness: 0.22,
  metalness: 0.06,
}});

const avatar = new THREE.Group();
scene.add(avatar);

// Torso Group
const torso = new THREE.Group();
torso.position.set(0, 0.95, 0);
avatar.add(torso);

// Fitted Modern Crewneck Shirt
const chestGeo = new THREE.CylinderGeometry(0.22, 0.18, 0.42, 28);
chestGeo.scale(1.10, 1.0, 0.88);
const chestMesh = new THREE.Mesh(chestGeo, shirtMat);
chestMesh.position.y = 0.10;
torso.add(chestMesh);

// Collar
const collarGeo = new THREE.TorusGeometry(0.088, 0.015, 16, 32);
collarGeo.scale(1.06, 0.94, 1.0);
const collarMesh = new THREE.Mesh(collarGeo, collarMat);
collarMesh.rotation.x = Math.PI / 2;
collarMesh.position.set(0, 0.30, 0);
torso.add(collarMesh);

// Neck
const neckGeo = new THREE.CylinderGeometry(0.062, 0.076, 0.14, 24);
const neckMesh = new THREE.Mesh(neckGeo, skinMat);
neckMesh.position.y = 0.36;
torso.add(neckMesh);

// Head Group
const headGroup = new THREE.Group();
headGroup.position.set(0, 0.51, 0);
torso.add(headGroup);

// Cranium
const headGeo = new THREE.SphereGeometry(0.162, 36, 36);
headGeo.scale(1.0, 1.18, 1.05);
const headMesh = new THREE.Mesh(headGeo, skinMat);
headGroup.add(headMesh);

// Smooth Chin
const chinGeo = new THREE.SphereGeometry(0.046, 24, 24);
chinGeo.scale(1.12, 0.88, 1.15);
const chinMesh = new THREE.Mesh(chinGeo, skinMat);
chinMesh.position.set(0, -0.115, 0.062);
headGroup.add(chinMesh);

// Rosy Cheeks
function createCheekBlush(isRight) {{
  const sign = isRight ? 1 : -1;
  const blushGeo = new THREE.SphereGeometry(0.036, 16, 16);
  blushGeo.scale(1.2, 0.7, 0.5);
  const mesh = new THREE.Mesh(blushGeo, blushMat);
  mesh.position.set(sign * 0.076, -0.01, 0.118);
  return mesh;
}}
headGroup.add(createCheekBlush(true));
headGroup.add(createCheekBlush(false));

// Smooth Human Nose
const noseBridgeGeo = new THREE.CylinderGeometry(0.009, 0.014, 0.052, 16);
const noseBridge = new THREE.Mesh(noseBridgeGeo, skinMat);
noseBridge.position.set(0, 0.026, 0.158);
noseBridge.rotation.x = -0.22;
headGroup.add(noseBridge);

const noseTipGeo = new THREE.SphereGeometry(0.0135, 20, 20);
const noseTip = new THREE.Mesh(noseTipGeo, skinMat);
noseTip.position.set(0, 0.006, 0.170);
headGroup.add(noseTip);

function createNostril(isRight) {{
  const sign = isRight ? 1 : -1;
  const geo = new THREE.SphereGeometry(0.0075, 12, 12);
  const mesh = new THREE.Mesh(geo, skinMat);
  mesh.position.set(sign * 0.012, 0.003, 0.162);
  return mesh;
}}
headGroup.add(createNostril(true));
headGroup.add(createNostril(false));

// Friendly Human Lips
const lipUpperGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.009, 20);
const lipUpper = new THREE.Mesh(lipUpperGeo, lipsMat);
lipUpper.rotation.z = Math.PI / 2;
lipUpper.scale.set(0.60, 1.0, 0.70);
lipUpper.position.set(0, -0.040, 0.156);
headGroup.add(lipUpper);

const lipLowerGeo = new THREE.SphereGeometry(0.016, 18, 18);
lipLowerGeo.scale(1.4, 0.65, 0.85);
const lipLower = new THREE.Mesh(lipLowerGeo, lipsMat);
lipLower.position.set(0, -0.054, 0.152);
headGroup.add(lipLower);

// Ears
function createEar(isRight) {{
  const sign = isRight ? 1 : -1;
  const earGroup = new THREE.Group();
  earGroup.position.set(sign * 0.162, 0.015, 0);

  const earOuter = new THREE.Mesh(new THREE.TorusGeometry(0.026, 0.008, 12, 20, Math.PI * 1.3), skinMat);
  earOuter.rotation.y = sign * (Math.PI / 2 + 0.15);
  earOuter.rotation.z = sign * 0.15;
  earGroup.add(earOuter);

  const earLobe = new THREE.Mesh(new THREE.SphereGeometry(0.011, 12, 12), skinMat);
  earLobe.position.set(0, -0.022, 0);
  earGroup.add(earLobe);

  return earGroup;
}}
headGroup.add(createEar(true));
headGroup.add(createEar(false));

// Modern Styled Hair
const hairVolumeGeo = new THREE.SphereGeometry(0.170, 32, 32);
hairVolumeGeo.scale(1.02, 1.12, 1.04);
const hairVolume = new THREE.Mesh(hairVolumeGeo, hairMat);
hairVolume.position.set(0, 0.050, -0.025);
headGroup.add(hairVolume);

const hairFrontGeo = new THREE.SphereGeometry(0.065, 16, 16);
hairFrontGeo.scale(1.8, 0.8, 0.9);
const hairFront = new THREE.Mesh(hairFrontGeo, hairMat);
hairFront.position.set(0.03, 0.135, 0.10);
hairFront.rotation.set(-0.2, 0.3, -0.15);
headGroup.add(hairFront);

// Eyes with Catchlights & Animated Eyelids
function createHumanEye(isRight) {{
  const sign = isRight ? 1 : -1;
  const eyeHolder = new THREE.Group();
  eyeHolder.position.set(sign * 0.052, 0.036, 0.138);

  const sclera = new THREE.Mesh(new THREE.SphereGeometry(0.020, 24, 24), eyeScleraMat);
  eyeHolder.add(sclera);

  const iris = new THREE.Mesh(new THREE.CircleGeometry(0.0125, 24), irisMat);
  iris.position.set(0, 0, 0.019);
  eyeHolder.add(iris);

  const catchlight = new THREE.Mesh(new THREE.SphereGeometry(0.0028, 8, 8), catchlightMat);
  catchlight.position.set(sign * 0.0035, 0.004, 0.0195);
  eyeHolder.add(catchlight);

  const upperLidGeo = new THREE.SphereGeometry(0.021, 20, 20, 0, Math.PI * 2, 0, Math.PI / 2);
  const upperLid = new THREE.Mesh(upperLidGeo, skinMat);
  upperLid.position.set(0, 0.003, 0);
  eyeHolder.add(upperLid);

  return {{ group: eyeHolder, upperLid: upperLid }};
}}

const leftEyeObj = createHumanEye(false);
const rightEyeObj = createHumanEye(true);
headGroup.add(leftEyeObj.group);
headGroup.add(rightEyeObj.group);

// Eyebrows
const leftBrow = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.009, 0.012), browMat);
leftBrow.position.set(-0.052, 0.074, 0.150);
headGroup.add(leftBrow);

const rightBrow = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.009, 0.012), browMat);
rightBrow.position.set(0.052, 0.074, 0.150);
headGroup.add(rightBrow);

// ============================================================================
// 4. ARTICULATED HUMAN ARMS & 3-PHALANGE SUPPLE HAND RIG
// ============================================================================
function buildHumanArm(isRight) {{
  const sign = isRight ? -1 : 1;
  const arm = {{ isRight: isRight, sign: sign }};

  arm.clavicle = new THREE.Group();
  arm.clavicle.position.set(sign * 0.26, 0.26, 0);
  torso.add(arm.clavicle);

  const deltoidMesh = new THREE.Mesh(new THREE.SphereGeometry(0.072, 24, 24), shirtMat);
  deltoidMesh.position.set(sign * 0.01, -0.02, 0);
  arm.clavicle.add(deltoidMesh);

  arm.upperArm = new THREE.Group();
  arm.clavicle.add(arm.upperArm);

  const upperMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.052, 0.045, 0.28, 20), shirtMat);
  upperMesh.position.y = -0.14;
  arm.upperArm.add(upperMesh);

  arm.elbow = new THREE.Group();
  arm.elbow.position.set(0, -0.28, 0);
  arm.upperArm.add(arm.elbow);

  const elbowMesh = new THREE.Mesh(new THREE.SphereGeometry(0.045, 20, 20), skinMat);
  arm.elbow.add(elbowMesh);

  arm.forearm = new THREE.Group();
  arm.elbow.add(arm.forearm);

  const foreMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.035, 0.25, 20), skinMat);
  foreMesh.position.y = -0.125;
  arm.forearm.add(foreMesh);

  arm.wrist = new THREE.Group();
  arm.wrist.position.set(0, -0.25, 0);
  arm.forearm.add(arm.wrist);

  const wristMesh = new THREE.Mesh(new THREE.SphereGeometry(0.034, 16, 16), skinMat);
  arm.wrist.add(wristMesh);

  arm.palm = new THREE.Group();
  arm.wrist.add(arm.palm);

  const palmMesh = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.082, 0.022), skinMat);
  palmMesh.position.y = -0.041;
  arm.palm.add(palmMesh);

  const thenarMesh = new THREE.Mesh(new THREE.SphereGeometry(0.022, 14, 14), skinMat);
  thenarMesh.position.set(-sign * 0.025, -0.025, 0.008);
  arm.palm.add(thenarMesh);

  arm.fingers = [];
  const fingerDefs = [
    {{ name: 'thumb',  x: -sign * 0.042, y: -0.020, z: 0.010, len: 0.048, rad: 0.0120, isThumb: true }},
    {{ name: 'index',  x: -sign * 0.026, y: -0.082, z: 0.001, len: 0.056, rad: 0.0105 }},
    {{ name: 'middle', x: -sign * 0.009, y: -0.086, z: 0.002, len: 0.062, rad: 0.0105 }},
    {{ name: 'ring',   x: sign * 0.009,  y: -0.082, z: 0.001, len: 0.055, rad: 0.0100 }},
    {{ name: 'pinky',  x: sign * 0.026,  y: -0.076, z: -0.001, len: 0.046, rad: 0.0090 }},
  ];

  fingerDefs.forEach((fd) => {{
    const fBase = new THREE.Group();
    fBase.position.set(fd.x, fd.y, fd.z);
    arm.palm.add(fBase);

    const knuckle = new THREE.Mesh(new THREE.SphereGeometry(fd.rad * 1.05, 12, 12), skinMat);
    fBase.add(knuckle);

    const baseLen = fd.len * 0.44;
    const pMesh = new THREE.Mesh(new THREE.CylinderGeometry(fd.rad, fd.rad * 0.92, baseLen, 12), skinMat);
    pMesh.position.y = -baseLen * 0.5;
    fBase.add(pMesh);

    const fMid = new THREE.Group();
    fMid.position.y = -baseLen;
    fBase.add(fMid);

    const midKnuckle = new THREE.Mesh(new THREE.SphereGeometry(fd.rad * 0.92, 10, 10), skinMat);
    fMid.add(midKnuckle);

    const midLen = fd.len * 0.32;
    const mMesh = new THREE.Mesh(new THREE.CylinderGeometry(fd.rad * 0.90, fd.rad * 0.80, midLen, 12), skinMat);
    mMesh.position.y = -midLen * 0.5;
    fMid.add(mMesh);

    const fTip = new THREE.Group();
    fTip.position.y = -midLen;
    fMid.add(fTip);

    const tipLen = fd.len * 0.24;
    const dMesh = new THREE.Mesh(new THREE.CylinderGeometry(fd.rad * 0.80, fd.rad * 0.60, tipLen, 12), skinMat);
    dMesh.position.y = -tipLen * 0.5;
    fTip.add(dMesh);

    const pulp = new THREE.Mesh(new THREE.SphereGeometry(fd.rad * 0.70, 10, 10), skinMat);
    pulp.position.set(0, -tipLen * 0.55, -fd.rad * 0.32);
    fTip.add(pulp);

    const nailGeo = new THREE.BoxGeometry(fd.rad * 1.15, tipLen * 0.65, 0.0035);
    const nail = new THREE.Mesh(nailGeo, nailMat);
    nail.position.set(0, -tipLen * 0.45, fd.rad * 0.58);
    fTip.add(nail);

    arm.fingers.push({{
      name: fd.name,
      base: fBase,
      mid: fMid,
      tip: fTip,
      isThumb: fd.isThumb || false
    }});
  }});

  return arm;
}}

const rightArm = buildHumanArm(true);
const leftArm = buildHumanArm(false);

// ============================================================================
// 5. ANATOMICAL KINEMATIC POSE MODEL & BIOMECHANICAL EASING
// ============================================================================
function createAnatomicalPose() {{
  return {{
    head: {{ rx: 0, ry: 0, rz: 0 }},
    brows: {{ y: 0, rot: 0 }},
    torso: {{ rx: 0, ry: 0, rz: 0 }},
    rArm: {{ clavY: 0, sElev: 20, sAbd: 16, sTwist: -10, eFlex: 80, fRoll: 35, wPitch: 0, wYaw: 0, wRoll: 0 }},
    rFingers: [
      {{ curl: 0.15, spread: 0.1 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.15, spread: -0.1 }}
    ],
    lArm: {{ clavY: 0, sElev: 20, sAbd: 16, sTwist: 10, eFlex: 80, fRoll: -35, wPitch: 0, wYaw: 0, wRoll: 0 }},
    lFingers: [
      {{ curl: 0.15, spread: -0.1 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.12, spread: 0 }},
      {{ curl: 0.15, spread: 0.1 }}
    ]
  }};
}}

const IDLE_POSE = createAnatomicalPose();
let currentPose = JSON.parse(JSON.stringify(IDLE_POSE));

function applyArmPose(arm, armData, fingersData) {{
  const sign = arm.sign;
  const deg2rad = Math.PI / 180;

  arm.clavicle.position.y = 0.26 + (armData.clavY || 0);

  arm.upperArm.rotation.set(
    -armData.sElev * deg2rad,
    sign * armData.sTwist * deg2rad,
    sign * armData.sAbd * deg2rad,
    'ZXY'
  );

  arm.elbow.rotation.x = -armData.eFlex * deg2rad;
  arm.forearm.rotation.y = sign * armData.fRoll * deg2rad;

  arm.wrist.rotation.set(
    -armData.wPitch * deg2rad,
    sign * (armData.wRoll || 0) * deg2rad,
    sign * armData.wYaw * deg2rad,
    'ZYX'
  );

  fingersData.forEach((f, idx) => {{
    const finger = arm.fingers[idx];
    if (finger) {{
      if (finger.isThumb) {{
        finger.base.rotation.y = sign * f.curl * Math.PI * 0.40;
        finger.base.rotation.x = -f.curl * Math.PI * 0.35;
        finger.base.rotation.z = -sign * (f.spread || 0.1);
        finger.mid.rotation.x  = -f.curl * Math.PI * 0.48;
        finger.tip.rotation.x  = -f.curl * Math.PI * 0.45;
      }} else {{
        const curlRad = f.curl * Math.PI * 0.54;
        finger.base.rotation.x = -curlRad * 0.50;
        finger.base.rotation.z = -sign * (f.spread || 0);
        finger.mid.rotation.x  = -curlRad * 0.64;
        finger.tip.rotation.x  = -curlRad * 0.42;
      }}
    }}
  }});
}}

function applyFullPose(p) {{
  headGroup.rotation.set(p.head.rx, p.head.ry, p.head.rz);

  const by = p.brows ? p.brows.y : 0;
  const br = p.brows ? p.brows.rot : 0;
  leftBrow.position.y = 0.074 + by;
  rightBrow.position.y = 0.074 + by;
  leftBrow.rotation.z = -br;
  rightBrow.rotation.z = br;

  torso.rotation.set(p.torso.rx, p.torso.ry, p.torso.rz);

  applyArmPose(rightArm, p.rArm, p.rFingers);
  applyArmPose(leftArm, p.lArm, p.lFingers);
}}

function lerp(a, b, t) {{
  return a + (b - a) * t;
}}

function lerpAnatomicalPose(p1, p2, t) {{
  const res = createAnatomicalPose();

  ['rx', 'ry', 'rz'].forEach(axis => {{
    res.head[axis] = lerp(p1.head[axis], p2.head[axis], t);
    res.torso[axis] = lerp(p1.torso[axis], p2.torso[axis], t);
  }});
  res.brows = {{
    y: lerp(p1.brows ? p1.brows.y : 0, p2.brows ? p2.brows.y : 0, t),
    rot: lerp(p1.brows ? p1.brows.rot : 0, p2.brows ? p2.brows.rot : 0, t)
  }};

  const armKeys = ['clavY', 'sElev', 'sAbd', 'sTwist', 'eFlex', 'fRoll', 'wPitch', 'wYaw', 'wRoll'];
  armKeys.forEach(k => {{
    res.rArm[k] = lerp(p1.rArm[k] || 0, p2.rArm[k] || 0, t);
    res.lArm[k] = lerp(p1.lArm[k] || 0, p2.lArm[k] || 0, t);
  }});

  for (let i = 0; i < 5; i++) {{
    res.rFingers[i] = {{
      curl: lerp(p1.rFingers[i].curl, p2.rFingers[i].curl, t),
      spread: lerp(p1.rFingers[i].spread || 0, p2.rFingers[i].spread || 0, t)
    }};
    res.lFingers[i] = {{
      curl: lerp(p1.lFingers[i].curl, p2.lFingers[i].curl, t),
      spread: lerp(p1.lFingers[i].spread || 0, p2.lFingers[i].spread || 0, t)
    }};
  }}
  return res;
}}

// ============================================================================
// 6. ALL 20 DISTINCT, UNMISTAKABLE & ACCURATE SIGN LANGUAGE GESTURES
// ============================================================================
const SIGNS = {{}};

function fingers(cT, cI, cM, cR, cP, sT=0.1, sI=0, sM=0, sR=0, sP=-0.1) {{
  return [
    {{ curl: cT, spread: sT }},
    {{ curl: cI, spread: sI }},
    {{ curl: cM, spread: sM }},
    {{ curl: cR, spread: sR }},
    {{ curl: cP, spread: sP }}
  ];
}}

// 1. HELLO: Temple salute reaching up, then waving hand side-to-side in the air
SIGNS['hello'] = [
  {{ duration: 420, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.02, sElev: 80, sAbd: 32, sTwist: -30, eFlex: 135, fRoll: 25, wPitch: 10, wYaw: -15 }};
    p.rFingers = fingers(0, 0, 0, 0, 0); // Flat B-hand
    p.head = {{ rx: -0.06, ry: -0.06, rz: 0 }};
    p.brows = {{ y: 0.015, rot: -0.08 }};
    return p;
  }})() }},
  {{ duration: 320, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.01, sElev: 75, sAbd: 48, sTwist: -10, eFlex: 90, fRoll: 10, wPitch: 0, wYaw: 25 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.2, 0.05, 0, -0.05, -0.15); // Open wave
    p.head = {{ rx: 0.04, ry: 0, rz: 0 }};
    p.brows = {{ y: 0.015, rot: -0.08 }};
    return p;
  }})() }},
  {{ duration: 320, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.01, sElev: 75, sAbd: 44, sTwist: -10, eFlex: 90, fRoll: 10, wPitch: 0, wYaw: -20 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.2, 0.05, 0, -0.05, -0.15);
    p.brows = {{ y: 0.015, rot: -0.08 }};
    return p;
  }})() }},
  {{ duration: 320, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.01, sElev: 75, sAbd: 48, sTwist: -10, eFlex: 90, fRoll: 10, wPitch: 0, wYaw: 20 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.2, 0.05, 0, -0.05, -0.15);
    return p;
  }})() }}
];

// 2. THANK YOU: Flat B-hand touches lips/chin, then sweeps FAR FORWARD & DOWN with palm facing up toward viewer
SIGNS['thank_you'] = SIGNS['thankyou'] = SIGNS['thanks'] = [
  {{ duration: 440, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 52, sAbd: 14, sTwist: -20, eFlex: 132, fRoll: 20, wPitch: 18, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0); // Flat fingertips on lips
    p.head = {{ rx: 0.06, ry: 0, rz: 0 }};
    p.brows = {{ y: 0.008, rot: 0 }};
    return p;
  }})() }},
  {{ duration: 750, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: -0.01, sElev: 38, sAbd: 15, sTwist: 5, eFlex: 25, fRoll: -55, wPitch: -20, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.15, 0.04, 0, -0.04, -0.1); // Palm flat up toward viewer!
    p.head = {{ rx: 0.18, ry: 0, rz: 0 }}; // Gracious bow
    p.torso = {{ rx: 0.08, ry: 0, rz: 0 }};
    p.brows = {{ y: 0.015, rot: 0 }};
    return p;
  }})() }}
];

// 3. PLEASE: Open flat hand rubs in a large, distinct clockwise circle on the center of the chest
SIGNS['please'] = [
  {{ duration: 380, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 32, sAbd: 12, sTwist: -30, eFlex: 112, fRoll: 55, wPitch: 22, wYaw: 8 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.18, 0.05, 0, -0.05, -0.1); // Wide open palm
    p.head = {{ rx: 0.06, ry: 0.04, rz: 0 }};
    p.torso = {{ rx: 0.04, ry: -0.04, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 400, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 38, sAbd: 20, sTwist: -15, eFlex: 104, fRoll: 42, wPitch: 15, wYaw: -12 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.18, 0.05, 0, -0.05, -0.1);
    p.head = {{ rx: 0.12, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 400, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: -0.01, sElev: 26, sAbd: 14, sTwist: -35, eFlex: 118, fRoll: 60, wPitch: 26, wYaw: 12 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.18, 0.05, 0, -0.05, -0.1);
    p.head = {{ rx: 0.08, ry: -0.04, rz: 0 }};
    return p;
  }})() }}
];

// 4. SORRY: Tight A-fist specifically rubbed over the HEART (left chest) with an apologetic head bow
SIGNS['sorry'] = [
  {{ duration: 440, pose: (() => {{
    const p = createAnatomicalPose();
    // Reaching across to the left chest over the heart
    p.rArm = {{ clavY: 0, sElev: 34, sAbd: 8, sTwist: -40, eFlex: 118, fRoll: 65, wPitch: 20, wYaw: 15 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1); // Tight closed fist
    p.head = {{ rx: 0.22, ry: 0.08, rz: 0.04 }}; // Sorrowful tilted bow
    p.torso = {{ rx: 0.08, ry: -0.06, rz: 0 }};
    p.brows = {{ y: -0.016, rot: 0.16 }}; // Furrowed apologetic brows
    return p;
  }})() }},
  {{ duration: 440, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.01, sElev: 38, sAbd: 14, sTwist: -32, eFlex: 110, fRoll: 55, wPitch: 14, wYaw: 5 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    p.head = {{ rx: 0.28, ry: 0.02, rz: 0 }};
    p.brows = {{ y: -0.018, rot: 0.18 }};
    return p;
  }})() }}
];

// 5. YES: Upright S-fist held forward like a person's head, nodding up and down sharply 2x, synchronized with head
SIGNS['yes'] = [
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 46, sAbd: 22, sTwist: -10, eFlex: 82, fRoll: 10, wPitch: -32, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1); // S-fist tilted back
    p.head = {{ rx: -0.10, ry: 0, rz: 0 }};
    p.brows = {{ y: 0.012, rot: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 46, sAbd: 22, sTwist: -10, eFlex: 82, fRoll: 10, wPitch: 35, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1); // S-fist nodded down sharply!
    p.head = {{ rx: 0.22, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 260, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 46, sAbd: 22, sTwist: -10, eFlex: 82, fRoll: 10, wPitch: -28, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    p.head = {{ rx: -0.08, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 46, sAbd: 22, sTwist: -10, eFlex: 82, fRoll: 10, wPitch: 32, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    p.head = {{ rx: 0.20, ry: 0, rz: 0 }};
    return p;
  }})() }}
];

// 6. NO: Index & middle fingers snap down hard onto thumb pad twice, head shakes side-to-side
SIGNS['no'] = [
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 52, sAbd: 26, sTwist: -10, eFlex: 92, fRoll: 10, wPitch: 0, wYaw: 0 }};
    // Wide open snapping beak
    p.rFingers = [{{ curl: 0.05, spread: 0.35 }}, {{ curl: 0, spread: 0.06 }}, {{ curl: 0, spread: -0.06 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: 0, ry: -0.22, rz: 0 }}; // Decisive side shake
    p.brows = {{ y: -0.012, rot: 0.12 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 52, sAbd: 26, sTwist: -10, eFlex: 92, fRoll: 10, wPitch: 0, wYaw: 0 }};
    // SNAP shut onto thumb!
    p.rFingers = [{{ curl: 0.85, spread: 0.1 }}, {{ curl: 0.85, spread: 0 }}, {{ curl: 0.85, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: 0, ry: 0.22, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 260, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 52, sAbd: 26, sTwist: -10, eFlex: 92, fRoll: 10, wPitch: 0, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.05, spread: 0.35 }}, {{ curl: 0, spread: 0.06 }}, {{ curl: 0, spread: -0.06 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: 0, ry: -0.18, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 52, sAbd: 26, sTwist: -10, eFlex: 92, fRoll: 10, wPitch: 0, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.85, spread: 0.1 }}, {{ curl: 0.85, spread: 0 }}, {{ curl: 0.85, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: 0, ry: 0.16, rz: 0 }};
    return p;
  }})() }}
];

// 7. HELP: Left hand flat palm up, Right hand thumbs-up fist placed on left palm, BOTH lift upward strongly together
SIGNS['help'] = [
  {{ duration: 460, pose: (() => {{
    const p = createAnatomicalPose();
    // Left hand: table platform at waist
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 12, sTwist: 20, eFlex: 90, fRoll: -88, wPitch: 0, wYaw: -5 }};
    p.lFingers = fingers(0, 0, 0, 0, 0); // Flat palm UP
    // Right hand: Thumbs-up resting directly on left palm
    p.rArm = {{ clavY: 0, sElev: 34, sAbd: 14, sTwist: -20, eFlex: 92, fRoll: 20, wPitch: 0, wYaw: 5 }};
    p.rFingers = [{{ curl: 0, spread: 0.50 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}]; // Big Thumbs-Up!
    p.head = {{ rx: 0.08, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 750, pose: (() => {{
    const p = createAnatomicalPose();
    // Both arms lift upward together toward viewer!
    p.lArm = {{ clavY: 0.03, sElev: 56, sAbd: 14, sTwist: 20, eFlex: 94, fRoll: -88, wPitch: 0, wYaw: -5 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0.03, sElev: 58, sAbd: 14, sTwist: -20, eFlex: 96, fRoll: 20, wPitch: 0, wYaw: 5 }};
    p.rFingers = [{{ curl: 0, spread: 0.50 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: -0.06, ry: 0, rz: 0 }}; // Encouraging uplifted gaze
    p.brows = {{ y: 0.015, rot: -0.05 }};
    return p;
  }})() }}
];

// 8. WATER: Iconic "W" handshape (Index, Middle, Ring up & spread) clearly tapping side of chin twice
SIGNS['water'] = [
  {{ duration: 380, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 56, sAbd: 20, sTwist: -18, eFlex: 126, fRoll: 15, wPitch: 12, wYaw: 0 }};
    // W-hand: thumb folds over pinky, other 3 stand tall & spread!
    p.rFingers = [{{ curl: 0.95, spread: 0.1 }}, {{ curl: 0, spread: 0.22 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.22 }}, {{ curl: 0.95, spread: 0 }}];
    p.head = {{ rx: 0.06, ry: -0.06, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.005, sElev: 50, sAbd: 20, sTwist: -18, eFlex: 115, fRoll: 15, wPitch: 8, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.95, spread: 0.1 }}, {{ curl: 0, spread: 0.22 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.22 }}, {{ curl: 0.95, spread: 0 }}];
    return p;
  }})() }},
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 56, sAbd: 20, sTwist: -18, eFlex: 126, fRoll: 15, wPitch: 12, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.95, spread: 0.1 }}, {{ curl: 0, spread: 0.22 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.22 }}, {{ curl: 0.95, spread: 0 }}];
    p.head = {{ rx: 0.06, ry: -0.06, rz: 0 }};
    return p;
  }})() }}
];

// 9. FOOD: Pinched fingertips (all 5 tips touching thumb) brought directly to mouth twice
SIGNS['food'] = [
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 54, sAbd: 14, sTwist: -22, eFlex: 130, fRoll: 12, wPitch: 18, wYaw: 0 }};
    // Pinched morsel handshape
    p.rFingers = fingers(0.60, 0.60, 0.60, 0.60, 0.60);
    p.head = {{ rx: 0.08, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.005, sElev: 46, sAbd: 14, sTwist: -22, eFlex: 114, fRoll: 12, wPitch: 10, wYaw: 0 }};
    p.rFingers = fingers(0.60, 0.60, 0.60, 0.60, 0.60);
    return p;
  }})() }},
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 54, sAbd: 14, sTwist: -22, eFlex: 130, fRoll: 12, wPitch: 18, wYaw: 0 }};
    p.rFingers = fingers(0.60, 0.60, 0.60, 0.60, 0.60);
    p.head = {{ rx: 0.08, ry: 0, rz: 0 }};
    return p;
  }})() }}
];

// 10. MEDICINE: Left hand flat palm up, Right middle finger bent down into center of palm and twisting like a pestle
SIGNS['medicine'] = [
  {{ duration: 440, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 12, sTwist: 20, eFlex: 90, fRoll: -88, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0); // Open palm UP
    // Right hand: middle finger bent down into palm
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 16, sTwist: -25, eFlex: 102, fRoll: 38, wPitch: 22, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.7, spread: 0.1 }}, {{ curl: 0.7, spread: 0 }}, {{ curl: 0.25, spread: 0 }}, {{ curl: 0.7, spread: 0 }}, {{ curl: 0.7, spread: 0 }}];
    p.head = {{ rx: 0.18, ry: 0, rz: 0 }}; // Looking down at mortar
    return p;
  }})() }},
  {{ duration: 460, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 12, sTwist: 20, eFlex: 90, fRoll: -88, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    // Twist pestle counter-clockwise!
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 16, sTwist: -25, eFlex: 102, fRoll: -28, wPitch: 22, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.7, spread: 0.1 }}, {{ curl: 0.7, spread: 0 }}, {{ curl: 0.25, spread: 0 }}, {{ curl: 0.7, spread: 0 }}, {{ curl: 0.7, spread: 0 }}];
    p.head = {{ rx: 0.18, ry: 0, rz: 0 }};
    return p;
  }})() }}
];

// 11. DOCTOR: Left inner wrist held up, Right 2-fingers tapping radial pulse twice with head cocked inquisitively
SIGNS['doctor'] = [
  {{ duration: 420, pose: (() => {{
    const p = createAnatomicalPose();
    // Left wrist exposed
    p.lArm = {{ clavY: 0, sElev: 30, sAbd: 10, sTwist: 25, eFlex: 92, fRoll: -75, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    // Right 2 fingers tapping pulse
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 16, sTwist: -25, eFlex: 105, fRoll: 25, wPitch: 25, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.8, spread: 0.1 }}, {{ curl: 0, spread: 0.02 }}, {{ curl: 0, spread: -0.02 }}, {{ curl: 0.9, spread: 0 }}, {{ curl: 0.9, spread: 0 }}];
    p.head = {{ rx: 0.16, ry: -0.08, rz: -0.05 }}; // Attentive doctor gaze
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 30, sAbd: 10, sTwist: 25, eFlex: 92, fRoll: -75, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0, sElev: 33, sAbd: 16, sTwist: -25, eFlex: 98, fRoll: 25, wPitch: 18, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.8, spread: 0.1 }}, {{ curl: 0, spread: 0.02 }}, {{ curl: 0, spread: -0.02 }}, {{ curl: 0.9, spread: 0 }}, {{ curl: 0.9, spread: 0 }}];
    return p;
  }})() }},
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 30, sAbd: 10, sTwist: 25, eFlex: 92, fRoll: -75, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 16, sTwist: -25, eFlex: 105, fRoll: 25, wPitch: 25, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.8, spread: 0.1 }}, {{ curl: 0, spread: 0.02 }}, {{ curl: 0, spread: -0.02 }}, {{ curl: 0.9, spread: 0 }}, {{ curl: 0.9, spread: 0 }}];
    p.head = {{ rx: 0.16, ry: -0.08, rz: -0.05 }};
    return p;
  }})() }}
];

// 12. FAMILY: Both hands in "F" shapes (circle ring with flared fingers), sweeping wide in circle and meeting at pinkies
SIGNS['family'] = [
  {{ duration: 480, pose: (() => {{
    const p = createAnatomicalPose();
    // F-hands touching index rings in center
    p.rArm = {{ clavY: 0, sElev: 38, sAbd: 12, sTwist: -15, eFlex: 90, fRoll: 20, wPitch: 0, wYaw: 0 }};
    p.lArm = {{ clavY: 0, sElev: 38, sAbd: 12, sTwist: 15, eFlex: 90, fRoll: -20, wPitch: 0, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.65, spread: 0.1 }}, {{ curl: 0.65, spread: 0 }}, {{ curl: 0, spread: 0.08 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.08 }}];
    p.lFingers = [{{ curl: 0.65, spread: 0.1 }}, {{ curl: 0.65, spread: 0 }}, {{ curl: 0, spread: 0.08 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.08 }}];
    p.head = {{ rx: 0.06, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 750, pose: (() => {{
    const p = createAnatomicalPose();
    // Sweep wide outward and meet pinkies together!
    p.rArm = {{ clavY: 0, sElev: 42, sAbd: 32, sTwist: -25, eFlex: 82, fRoll: -35, wPitch: 0, wYaw: 0 }};
    p.lArm = {{ clavY: 0, sElev: 42, sAbd: 32, sTwist: 25, eFlex: 82, fRoll: 35, wPitch: 0, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.65, spread: 0.1 }}, {{ curl: 0.65, spread: 0 }}, {{ curl: 0, spread: 0.08 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.08 }}];
    p.lFingers = [{{ curl: 0.65, spread: 0.1 }}, {{ curl: 0.65, spread: 0 }}, {{ curl: 0, spread: 0.08 }}, {{ curl: 0, spread: 0 }}, {{ curl: 0, spread: -0.08 }}];
    p.head = {{ rx: 0.10, ry: 0, rz: 0 }};
    return p;
  }})() }}
];

// 13. WORK: Both hands in S-fists. Right fist hammers down firmly onto back of left wrist twice
SIGNS['work'] = [
  {{ duration: 380, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -30, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0.95, 1, 1, 1, 1);
    // Right fist raised high ready to strike
    p.rArm = {{ clavY: 0.02, sElev: 52, sAbd: 16, sTwist: -20, eFlex: 105, fRoll: 25, wPitch: 15, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    p.head = {{ rx: 0.14, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 280, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -30, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0.95, 1, 1, 1, 1);
    // Strike down on left wrist!
    p.rArm = {{ clavY: -0.01, sElev: 36, sAbd: 12, sTwist: -20, eFlex: 88, fRoll: 20, wPitch: 0, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    return p;
  }})() }},
  {{ duration: 340, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -30, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0.95, 1, 1, 1, 1);
    p.rArm = {{ clavY: 0.02, sElev: 50, sAbd: 16, sTwist: -20, eFlex: 102, fRoll: 25, wPitch: 15, wYaw: 0 }};
    p.rFingers = fingers(0.95, 1, 1, 1, 1);
    return p;
  }})() }}
];

// 14. SCHOOL: Left hand palm up, Right hand palm down, clapping down firmly twice
SIGNS['school'] = [
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -85, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0); // Flat palm UP
    // Right hand raised above palm
    p.rArm = {{ clavY: 0.02, sElev: 56, sAbd: 16, sTwist: -20, eFlex: 105, fRoll: 75, wPitch: 12, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0); // Flat palm DOWN
    p.head = {{ rx: 0.12, ry: 0, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 260, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -85, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    // CLAP down onto left palm!
    p.rArm = {{ clavY: -0.01, sElev: 35, sAbd: 12, sTwist: -20, eFlex: 88, fRoll: 75, wPitch: 0, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0);
    return p;
  }})() }},
  {{ duration: 340, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 32, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -85, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0.02, sElev: 54, sAbd: 16, sTwist: -20, eFlex: 102, fRoll: 75, wPitch: 12, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0);
    return p;
  }})() }}
];

// 15. HOME: Pinched flat-O touches mouth, then glides back across cheek to touch ear
SIGNS['home'] = [
  {{ duration: 420, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.012, sElev: 50, sAbd: 16, sTwist: -20, eFlex: 124, fRoll: 18, wPitch: 12, wYaw: 0 }};
    p.rFingers = fingers(0.6, 0.6, 0.6, 0.6, 0.6); // Touch near mouth
    p.head = {{ rx: 0.06, ry: -0.06, rz: 0 }};
    return p;
  }})() }},
  {{ duration: 580, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.02, sElev: 68, sAbd: 32, sTwist: -28, eFlex: 138, fRoll: 25, wPitch: 15, wYaw: 0 }};
    p.rFingers = fingers(0.6, 0.6, 0.6, 0.6, 0.6); // Glide to ear!
    p.head = {{ rx: -0.02, ry: -0.10, rz: 0 }};
    return p;
  }})() }}
];

// 16. MONEY: Left hand flat palm up, Right hand rubbing thumb across index & middle tips in cash counting motion
SIGNS['money'] = [
  {{ duration: 400, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 30, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -85, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 15, sTwist: -20, eFlex: 94, fRoll: 35, wPitch: 10, wYaw: 0 }};
    p.rFingers = [{{ curl: 0.1, spread: 0.25 }}, {{ curl: 0.35, spread: 0 }}, {{ curl: 0.35, spread: 0 }}, {{ curl: 0.95, spread: 0 }}, {{ curl: 0.95, spread: 0 }}];
    p.head = {{ rx: 0.16, ry: 0, rz: 0 }}; // Looking at money
    return p;
  }})() }},
  {{ duration: 400, pose: (() => {{
    const p = createAnatomicalPose();
    p.lArm = {{ clavY: 0, sElev: 30, sAbd: 10, sTwist: 20, eFlex: 90, fRoll: -85, wPitch: 0, wYaw: 0 }};
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.rArm = {{ clavY: 0, sElev: 36, sAbd: 15, sTwist: -20, eFlex: 94, fRoll: 35, wPitch: 10, wYaw: 0 }};
    // Rubbing thumb across fingertips!
    p.rFingers = [{{ curl: 0.65, spread: 0.1 }}, {{ curl: 0.75, spread: 0 }}, {{ curl: 0.75, spread: 0 }}, {{ curl: 0.95, spread: 0 }}, {{ curl: 0.95, spread: 0 }}];
    return p;
  }})() }}
];

// 17. PHONE: "Y" handshape (thumb & pinky flared) held to ear with head cocked in phone conversation
SIGNS['phone'] = [
  {{ duration: 650, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.02, sElev: 76, sAbd: 32, sTwist: -30, eFlex: 142, fRoll: 28, wPitch: 10, wYaw: 0 }};
    // Iconic Y-handshape
    p.rFingers = [{{ curl: 0, spread: 0.50 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 0, spread: -0.40 }}];
    p.head = {{ rx: 0.02, ry: -0.08, rz: -0.14 }}; // Comfortable phone tilt
    p.brows = {{ y: 0.01, rot: 0 }};
    return p;
  }})() }}
];

// 18. HAPPY: Both open hands brush upward against chest twice in joyful sweeps with beaming smile
SIGNS['happy'] = [
  {{ duration: 360, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 28, sAbd: 16, sTwist: -25, eFlex: 105, fRoll: 45, wPitch: 15, wYaw: 0 }};
    p.lArm = {{ clavY: 0, sElev: 28, sAbd: 16, sTwist: 25, eFlex: 105, fRoll: -45, wPitch: 15, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0);
    p.lFingers = fingers(0, 0, 0, 0, 0);
    p.head = {{ rx: 0.02, ry: 0, rz: 0 }};
    p.brows = {{ y: 0.02, rot: -0.08 }};
    return p;
  }})() }},
  {{ duration: 440, pose: (() => {{
    const p = createAnatomicalPose();
    // Joyful upward sweep!
    p.rArm = {{ clavY: 0.02, sElev: 56, sAbd: 28, sTwist: -20, eFlex: 118, fRoll: 30, wPitch: -5, wYaw: 0 }};
    p.lArm = {{ clavY: 0.02, sElev: 56, sAbd: 28, sTwist: 20, eFlex: 118, fRoll: -30, wPitch: -5, wYaw: 0 }};
    p.rFingers = fingers(0, 0, 0, 0, 0, 0.15, 0.04, 0, -0.04, -0.1);
    p.lFingers = fingers(0, 0, 0, 0, 0, 0.15, 0.04, 0, -0.04, -0.1);
    p.head = {{ rx: -0.10, ry: 0, rz: 0 }}; // Uplifted face!
    p.brows = {{ y: 0.025, rot: -0.10 }};
    return p;
  }})() }}
];

// 19. SAD: Both open hands held in front of face, then slowly draw DOWNWARD as head bows in sorrow
SIGNS['sad'] = [
  {{ duration: 450, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0.015, sElev: 60, sAbd: 20, sTwist: -20, eFlex: 128, fRoll: 10, wPitch: 0, wYaw: 0 }};
    p.lArm = {{ clavY: 0.015, sElev: 60, sAbd: 20, sTwist: 20, eFlex: 128, fRoll: -10, wPitch: 0, wYaw: 0 }};
    p.rFingers = fingers(0.3, 0.3, 0.3, 0.3, 0.3);
    p.lFingers = fingers(0.3, 0.3, 0.3, 0.3, 0.3);
    p.head = {{ rx: 0.04, ry: 0, rz: 0 }};
    p.brows = {{ y: -0.014, rot: 0.14 }};
    return p;
  }})() }},
  {{ duration: 750, pose: (() => {{
    const p = createAnatomicalPose();
    // Hands draw down to waist, head drops in deep sadness
    p.rArm = {{ clavY: -0.015, sElev: 26, sAbd: 14, sTwist: -10, eFlex: 80, fRoll: 15, wPitch: 12, wYaw: 0 }};
    p.lArm = {{ clavY: -0.015, sElev: 26, sAbd: 14, sTwist: 10, eFlex: 80, fRoll: -15, wPitch: 12, wYaw: 0 }};
    p.rFingers = fingers(0.45, 0.45, 0.45, 0.45, 0.45);
    p.lFingers = fingers(0.45, 0.45, 0.45, 0.45, 0.45);
    p.head = {{ rx: 0.28, ry: 0, rz: 0 }};
    p.torso = {{ rx: 0.08, ry: 0, rz: 0 }};
    p.brows = {{ y: -0.018, rot: 0.18 }};
    return p;
  }})() }}
];

// 20. PAIN: Both index fingers point directly at each other, twisting and jabbing inward sharply with a facial wince
SIGNS['pain'] = [
  {{ duration: 380, pose: (() => {{
    const p = createAnatomicalPose();
    // Index fingers pointing at each other in chest
    p.rArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: -15, eFlex: 92, fRoll: 40, wPitch: 10, wYaw: -12 }};
    p.lArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: 15, eFlex: 92, fRoll: -40, wPitch: 10, wYaw: 12 }};
    p.rFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.lFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: -0.08, ry: 0.04, rz: 0 }};
    p.brows = {{ y: -0.014, rot: 0.14 }}; // Pain wince
    return p;
  }})() }},
  {{ duration: 320, pose: (() => {{
    const p = createAnatomicalPose();
    // Sharp twist & jab!
    p.rArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: -15, eFlex: 92, fRoll: -30, wPitch: 10, wYaw: -12 }};
    p.lArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: 15, eFlex: 92, fRoll: 30, wPitch: 10, wYaw: 12 }};
    p.rFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.lFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: -0.04, ry: -0.04, rz: 0 }};
    p.brows = {{ y: -0.016, rot: 0.16 }};
    return p;
  }})() }},
  {{ duration: 380, pose: (() => {{
    const p = createAnatomicalPose();
    p.rArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: -15, eFlex: 92, fRoll: 40, wPitch: 10, wYaw: -12 }};
    p.lArm = {{ clavY: 0, sElev: 38, sAbd: 18, sTwist: 15, eFlex: 92, fRoll: -40, wPitch: 10, wYaw: 12 }};
    p.rFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.lFingers = [{{ curl: 0.9, spread: 0.1 }}, {{ curl: 0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}, {{ curl: 1.0, spread: 0 }}];
    p.head = {{ rx: -0.06, ry: 0, rz: 0 }};
    return p;
  }})() }}
];

// ============================================================================
// 7. ANIMATION PLAYBACK & SMOOTH BIOMECHANICAL EASING
// ============================================================================
let animQueue = [];
let currentStepIdx = 0;
let stepStartTime = 0;
let stepDuration = 500;
let prevStepPose = IDLE_POSE;
let nextStepPose = IDLE_POSE;
let isPlaying = false;
let currentActiveSign = "";
let speedMultiplier = 1.0;

const statusBadge = document.getElementById('statusBadge');
const speedSelect = document.getElementById('speedSelect');

function playSign(signKey) {{
  const key = signKey.toLowerCase().replace(/ /g, '_');
  currentActiveSign = key;

  // Highlight active chip
  document.querySelectorAll('.chip').forEach(c => {{
    c.classList.toggle('active', c.getAttribute('data-sign') === key);
  }});

  const keyframes = SIGNS[key] || SIGNS['hello'];
  const formattedName = key.replace(/_/g, ' ').toUpperCase();
  statusBadge.innerText = 'Signing: ' + formattedName;
  statusBadge.style.background = 'linear-gradient(135deg, #10B981, #06B6D4)';

  const seq = [];
  keyframes.forEach(kf => {{
    seq.push({{ duration: kf.duration, pose: kf.pose }});
  }});
  // Return smoothly to idle stance
  seq.push({{ duration: 680, pose: IDLE_POSE }});

  animQueue = seq;
  currentStepIdx = 0;
  prevStepPose = currentPose;
  nextStepPose = animQueue[0].pose;
  stepDuration = animQueue[0].duration / speedMultiplier;
  stepStartTime = performance.now();
  isPlaying = true;
}}

window.selectSign = function(signKey) {{
  playSign(signKey);
}};

speedSelect.addEventListener('change', (e) => {{
  speedMultiplier = parseFloat(e.target.value) || 1.0;
}});

// ============================================================================
// 8. RENDER LOOP, REALISTIC BLINKING & MINIMUM-JERK POLYNOMIAL
// ============================================================================
let clock = new THREE.Clock();
let nextBlinkTime = 2.0;
let isBlinking = false;
let blinkStartTime = 0;

function animate() {{
  requestAnimationFrame(animate);

  const now = performance.now();
  const elapsed = clock.getElapsedTime();

  // Natural breathing rhythm
  const breath = Math.sin(elapsed * 2.3) * 0.010;
  torso.position.y = 0.95 + breath;
  chestMesh.scale.set(1.10 + breath * 0.15, 1, 0.88 + breath * 0.15);

  // Human Eye Blinking
  if (!isBlinking && elapsed > nextBlinkTime) {{
    isBlinking = true;
    blinkStartTime = elapsed;
  }}
  if (isBlinking) {{
    const blinkT = (elapsed - blinkStartTime) / 0.14;
    if (blinkT <= 1.0) {{
      const eyelidScale = Math.sin(blinkT * Math.PI);
      leftEyeObj.upperLid.position.y = 0.003 - eyelidScale * 0.015;
      rightEyeObj.upperLid.position.y = 0.003 - eyelidScale * 0.015;
    }} else {{
      leftEyeObj.upperLid.position.y = 0.003;
      rightEyeObj.upperLid.position.y = 0.003;
      isBlinking = false;
      nextBlinkTime = elapsed + 3.0 + Math.random() * 2.5;
    }}
  }}

  // Animation Interpolation using Minimum-Jerk Biomechanical Polynomial:
  // f(t) = 10*t^3 - 15*t^4 + 6*t^5
  if (isPlaying && animQueue.length > 0) {{
    const tNorm = Math.min(1.0, (now - stepStartTime) / stepDuration);
    const easedT = 10 * Math.pow(tNorm, 3) - 15 * Math.pow(tNorm, 4) + 6 * Math.pow(tNorm, 5);

    currentPose = lerpAnatomicalPose(prevStepPose, nextStepPose, easedT);
    applyFullPose(currentPose);

    if (tNorm >= 1.0) {{
      currentStepIdx++;
      if (currentStepIdx < animQueue.length) {{
        prevStepPose = nextStepPose;
        nextStepPose = animQueue[currentStepIdx].pose;
        stepDuration = animQueue[currentStepIdx].duration / speedMultiplier;
        stepStartTime = performance.now();
      }} else {{
        isPlaying = false;
        statusBadge.innerText = 'Ready';
        statusBadge.style.background = 'linear-gradient(135deg, #10B981, #06B6D4)';
      }}
    }}
  }} else {{
    // Idle Organic Micro-Balance
    const idleWithDrift = lerpAnatomicalPose(currentPose, IDLE_POSE, 0.08);
    idleWithDrift.head.rx = Math.sin(elapsed * 1.5) * 0.02;
    idleWithDrift.head.ry = Math.cos(elapsed * 1.1) * 0.03;
    applyFullPose(idleWithDrift);
    currentPose = idleWithDrift;
  }}

  controls.update();
  renderer.render(scene, camera);
}}

window.addEventListener('resize', () => {{
  if (!container || !renderer || !camera) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}});

animate();

// Initial sign play
const initialSign = {safe_sign};
setTimeout(() => {{
  playSign(initialSign || 'hello');
}}, 600);
</script>
</body>
</html>
"""


def render_3d_avatar(sign: str = "hello", height: int = 580) -> None:
    """Render the 3D Sign Language Avatar in Streamlit."""
    html_content = get_3d_avatar_html(sign)
    components.html(html_content, height=height, scrolling=False)
