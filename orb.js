/* ================================================================
   WebGL Orb Renderer — ported from React/OGL to vanilla JS
   ================================================================
   This file renders the glowing, animated orb that serves as the
   visual centerpiece / background element of the JARVIS AI assistant
   UI. The orb is drawn entirely on the GPU using WebGL and GLSL
   shaders — no images or SVGs are involved.
   HOW IT WORKS (high-level):
   1. A full-screen <canvas> is created inside a container element.
   2. A WebGL context is obtained on that canvas.
   3. A vertex shader positions a single full-screen triangle, and a
      fragment shader runs *per pixel* to compute the orb's color
      using 3D simplex noise, hue-shifting math, and procedural
      lighting.
   4. An animation loop (requestAnimationFrame) feeds the shader a
      steadily increasing time value each frame, which makes the orb
      swirl, pulse, and react to state changes (e.g. "speaking").
   KEY CONCEPTS FOR LEARNERS:
   - **Vertex shader**: runs once per vertex. Here it just maps our
     triangle so it covers the whole screen.
   - **Fragment shader**: runs once per *pixel*. This is where all the
     visual magic happens — noise, lighting, color mixing.
   - **Uniforms**: values we send from JavaScript into the shader each
     frame (time, resolution, color settings, etc.).
   - **Simplex noise** (snoise3): a smooth random function that gives
     the orb its organic, cloud-like movement.
   The class exposes a simple API:
     new OrbRenderer(containerEl, options)   – start rendering
     .setActive(true/false)                  – pulse the orb (e.g. TTS speaking)
     .destroy()                              – tear everything down
   ================================================================ */

class OrbRenderer {
   /**
    * Creates a new OrbRenderer and immediately begins animating.
    *
    * @param {HTMLElement} container  – the DOM element the canvas will fill.
    * @param {Object}      opts      – optional tweaks:
    *   @param {number}   opts.hue             – base hue rotation in degrees (default 0).
    *   @param {number}   opts.hoverIntensity  – strength of the wavy hover/active distortion (default 0.2).
    *   @param {number[]} opts.backgroundColor – RGB triplet [r,g,b] each 0-1 (default dark navy).
    */
   constructor(container, opts = {}) {
      this.container = container;
      this.hue = opts.hue ?? 0;
      this.hoverIntensity = opts.hoverIntensity ?? 0.2;
      this.bgColor = opts.backgroundColor ?? [0.02, 0.02, 0.06];

      // Animation state — these are smoothly interpolated each frame
      // to avoid jarring jumps when setActive() is called.
      this.targetHover = 0;   // where we want hover to be (0 or 1)
      this.currentHover = 0;  // smoothly chases targetHover
      this.currentRot = 0;    // cumulative rotation (radians) applied while active
      this.lastTs = 0;        // timestamp of previous frame for delta-time calculation

      // SPEAKING MODE — a distinct animation pattern (not just "active")
      // used while TYRON is speaking aloud, so it visually reads as
      // "talking" rather than just "processing/active". Layers a few
      // irregular sine waves at different speeds to fake a talk-burst
      // rhythm (like real speech amplitude, not a smooth breathing pulse),
      // plus a subtle hue shimmer synced to the same rhythm.
      this.speaking = false;
      this.speakClock = 0;  // smoothly chases targetHover
      this.currentRot = 0;    // cumulative rotation (radians) applied while active
      this.lastTs = 0;        // timestamp of previous frame for delta-time calculation

      // Create and insert the drawing surface
      this.canvas = document.createElement('canvas');
      this.canvas.style.width = '100%';
      this.canvas.style.height = '100%';
      this.container.appendChild(this.canvas);

      // Acquire a WebGL 1 context.
      // alpha:true lets the orb float over whatever is behind the canvas.
      // premultipliedAlpha:false keeps our alpha blending straightforward.
      this.gl = this.canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false, antialias: false });
      if (!this.gl) { console.warn('WebGL not available'); return; }

      // Compile shaders, create buffers, look up uniform locations
      this._build();
      // Set the canvas resolution to match its CSS size × devicePixelRatio
      this._resize();
      // Re-adjust whenever the browser window changes size
      this._onResize = this._resize.bind(this);
      window.addEventListener('resize', this._onResize);
      // Kick off the animation loop
      this._raf = requestAnimationFrame(this._loop.bind(this));
   }

   /* =============================================================
      VERTEX SHADER (GLSL)
      =============================================================
      The vertex shader runs once for each vertex we send to the GPU
      (in our case just 3 — a single triangle that covers the whole
      screen).
      Inputs (attributes):
        position – the XY clip-space coordinate of this vertex.
        uv       – a texture coordinate we pass through to the
                    fragment shader so it knows where on the
                    "screen rectangle" each pixel is.
      Output:
        gl_Position – the final clip-space position (vec4).
        vUv         – passed to the fragment shader via a "varying".
      ============================================================= */
   static VERT = `
    precision highp float;
    attribute vec2 position;
    attribute vec2 uv;
    varying vec2 vUv;
    void main(){ vUv=uv; gl_Position=vec4(position,0.0,1.0); }`;

   /* =============================================================
      FRAGMENT SHADER (GLSL)
      =============================================================
      The fragment shader runs once for every pixel on screen. It
      receives the interpolated UV coordinate from the vertex shader
      and computes the final RGBA color for that pixel.
      UNIFORMS (values supplied from JavaScript every frame):
        iTime           – elapsed time in seconds; drives all animation.
        iResolution     – vec3(canvasWidth, canvasHeight, aspectRatio).
        hue             – degree offset applied to the base palette via
                          YIQ color-space rotation (lets you recolor the
                          whole orb without changing any other code).
        hover           – 0.0 → 1.0 interpolation: how "active" the orb
                          is right now. Drives the wavy UV distortion.
        rot             – current rotation angle (radians). Accumulated
                          on the JS side while the orb is active.
        hoverIntensity  – multiplier for the wavy UV distortion amplitude.
        backgroundColor – the scene's background color (RGB 0-1). The
                          shader blends toward this so the orb sits
                          naturally on any background.
      The shader contains several helper functions (explained inline
      below) and a main draw() routine that assembles the orb.
      ============================================================= */
   static FRAG = `
    precision highp float;
    uniform float iTime;
    uniform vec3  iResolution;
    uniform float hue;
    uniform float hover;
    uniform float rot;
    uniform float hoverIntensity;
    uniform vec3  backgroundColor;
    varying vec2  vUv;

    /* ----- Color-space conversion: RGB ↔ YIQ ----- */
    vec3 rgb2yiq(vec3 c){float y=dot(c,vec3(.299,.587,.114));float i=dot(c,vec3(.596,-.274,-.322));float q=dot(c,vec3(.211,-.523,.312));return vec3(y,i,q);}
    vec3 yiq2rgb(vec3 c){return vec3(c.x+.956*c.y+.621*c.z,c.x-.272*c.y-.647*c.z,c.x-1.106*c.y+1.703*c.z);}
    vec3 adjustHue(vec3 color,float hueDeg){float h=hueDeg*3.14159265/180.0;vec3 yiq=rgb2yiq(color);float cosA=cos(h);float sinA=sin(h);float i2=yiq.y*cosA-yiq.z*sinA;float q2=yiq.y*sinA+yiq.z*cosA;yiq.y=i2;yiq.z=q2;return yiq2rgb(yiq);}

    /* ----- 3D Simplex Noise (snoise3) & FBM ----- */
    vec3 hash33(vec3 p3){p3=fract(p3*vec3(.1031,.11369,.13787));p3+=dot(p3,p3.yxz+19.19);return -1.0+2.0*fract(vec3(p3.x+p3.y,p3.x+p3.z,p3.y+p3.z)*p3.zyx);}
    float snoise3(vec3 p){const float K1=.333333333;const float K2=.166666667;vec3 i=floor(p+(p.x+p.y+p.z)*K1);vec3 d0=p-(i-(i.x+i.y+i.z)*K2);vec3 e=step(vec3(0.0),d0-d0.yzx);vec3 i1=e*(1.0-e.zxy);vec3 i2=1.0-e.zxy*(1.0-e);vec3 d1=d0-(i1-K2);vec3 d2=d0-(i2-K1);vec3 d3=d0-0.5;vec4 h=max(0.6-vec4(dot(d0,d0),dot(d1,d1),dot(d2,d2),dot(d3,d3)),0.0);vec4 n=h*h*h*h*vec4(dot(d0,hash33(i)),dot(d1,hash33(i+i1)),dot(d2,hash33(i+i2)),dot(d3,hash33(i+1.0)));return dot(vec4(31.316),n);}

    float fbm(vec3 p) {
        float v = 0.0;
        float a = 0.5;
        vec3 shift = vec3(100.0);
        for (int i = 0; i < 4; ++i) {
            v += a * snoise3(p);
            p = p * 2.02 + shift;
            a *= 0.5;
        }
        return v;
    }

    vec4 extractAlpha(vec3 c){float a=max(max(c.r,c.g),c.b);return vec4(c/(a+1e-5),a);}

    /* ----- Palette & geometry constants ----- */
    const vec3 baseColor1=vec3(.55, .15, 1.0);     // deep electric violet
    const vec3 baseColor2=vec3(.00, .88, 1.0);     // brilliant electric cyan
    const vec3 baseColor3=vec3(.05, .04, .50);     // deep indigo core
    const float innerRadius=0.56;   // tighter inner core for more glow density
    const float noiseScale=0.68;   // noise zoom scale

    float light1(float i,float a,float d){return i/(1.0+d*a);}
    float light2(float i,float a,float d){return i/(1.0+d*d*a);}

    vec4 draw(vec2 uv){
        vec3 c1=adjustHue(baseColor1,hue);
        vec3 c2=adjustHue(baseColor2,hue);
        vec3 c3=adjustHue(baseColor3,hue);

        float ang=atan(uv.y,uv.x);
        float len=length(uv);
        float invLen=len>0.0?1.0/len:0.0;
        float bgLum=dot(backgroundColor,vec3(.299,.587,.114));

        // Organic multi-layered noise for fluid blob contours
        float n1 = fbm(vec3(uv * noiseScale, iTime * 0.45)) * 0.5 + 0.5;
        float n2 = snoise3(vec3(uv * (noiseScale * 1.8), iTime * 0.75 + ang * 0.3)) * 0.5 + 0.5;
        float n0 = mix(n1, n2, 0.35 + hover * 0.25);

        // Fluid morphing radius
        float dynamicRadius = mix(innerRadius, 0.95, n0);
        float r0 = dynamicRadius + sin(ang * 4.0 + iTime * 1.5) * (0.03 + hover * hoverIntensity * 0.05);

        float d0=distance(uv,(r0*invLen)*uv);
        float v0=light1(1.0, 8.0 + hover * 4.0, d0);
        v0*=smoothstep(r0*1.12, r0*0.85, len);

        float innerFade=smoothstep(r0*0.6, r0*0.95, len);
        v0*=mix(innerFade, 1.0, bgLum*0.7);

        // Swirling angular color blend
        float cl=cos(ang + iTime * 1.5 + n0 * 2.0) * 0.5 + 0.5;
        float a2=iTime * -0.8;
        vec2 pos=vec2(cos(a2), sin(a2)) * r0;
        float d=distance(uv, pos);

        float v1=light2(1.8 + hover * 0.6, 4.0, d);
        v1*=light1(1.0, 40.0, d0);

        float v2=smoothstep(1.05, mix(innerRadius, 1.0, n0 * 0.5), len);
        float v3=smoothstep(innerRadius * 0.8, mix(innerRadius, 1.0, 0.5), len);

        vec3 colBase=mix(c1, c2, cl);
        colBase += vec3(0.10, 0.45, 0.60) * pow(v0, 1.8); // vivid cyan core glow
        colBase += vec3(0.35, 0.05, 0.55) * pow(v1, 2.5) * hover; // violet plasma burst on active

        float fadeAmt=mix(1.0, 0.1, bgLum);
        vec3 darkCol=mix(c3, colBase, v0);
        darkCol=(darkCol + v1) * v2 * v3;
        darkCol=clamp(darkCol, 0.0, 1.0);

        vec3 lightCol=(colBase + v1) * mix(1.0, v2 * v3, fadeAmt);
        lightCol=mix(backgroundColor, lightCol, v0);
        lightCol=clamp(lightCol, 0.0, 1.0);

        vec3 fc=mix(darkCol, lightCol, bgLum);
        return extractAlpha(fc);
    }

    vec4 mainImage(vec2 fragCoord){
        vec2 center=iResolution.xy*0.5;
        float sz=min(iResolution.x, iResolution.y);
        vec2 uv=(fragCoord-center)/sz*2.0;

        // Smooth rotation
        float s2=sin(rot);
        float c2=cos(rot);
        uv=vec2(c2*uv.x-s2*uv.y, s2*uv.x+c2*uv.y);

        // Liquid domain warping distortion driven by active/speaking state
        float warpFactor = (hover * hoverIntensity * 0.14) + 0.035;
        float waveX = sin(uv.y * 5.0 + iTime * 1.6) * cos(uv.x * 3.5 + iTime * 1.1);
        float waveY = cos(uv.x * 5.0 + iTime * 1.6) * sin(uv.y * 3.5 + iTime * 1.1);

        uv.x += waveX * warpFactor;
        uv.y += waveY * warpFactor;

        return draw(uv);
    }

    void main(){
        vec2 fc=vUv*iResolution.xy;
        vec4 col=mainImage(fc);
        gl_FragColor=vec4(col.rgb*col.a, col.a);
    }`;

   /* =============================================================
      _compile(type, src)
      =============================================================
      Compiles a single GLSL shader (vertex or fragment).
      WebGL shaders are written in GLSL (a C-like language) and must
      be compiled at runtime by the GPU driver. If compilation fails
      (e.g. syntax error in the GLSL), we log the error and return
      null so _build() can bail out gracefully.
      ============================================================= */
   _compile(type, src) {
      const gl = this.gl;
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
         console.error('Shader compile error:', gl.getShaderInfoLog(s));
         gl.deleteShader(s);
         return null;
      }
      return s;
   }

   /* =============================================================
      _build()
      =============================================================
      Sets up everything the GPU needs to render the orb:
      1. COMPILE both shaders (vertex + fragment).
      2. LINK them into a "program" — the GPU pipeline that will run
         every frame.
      3. CREATE VERTEX BUFFERS. We use a single oversized triangle
         (the "full-screen triangle" trick) instead of a quad. Its 3
         vertices at (-1,-1), (3,-1), (-1,3) in clip space cover the
         entire [-1,1]² viewport and beyond, so every pixel gets a
         fragment shader invocation. This is faster than two triangles
         because the GPU only processes one primitive.
      4. LOOK UP UNIFORM LOCATIONS. gl.getUniformLocation returns a
         handle we use each frame to send updated values to the shader.
      5. ENABLE ALPHA BLENDING so the orb composites transparently
         over whatever is behind the canvas.
      ============================================================= */
   _build() {
      const gl = this.gl;
      const vs = this._compile(gl.VERTEX_SHADER, OrbRenderer.VERT);
      const fs = this._compile(gl.FRAGMENT_SHADER, OrbRenderer.FRAG);
      if (!vs || !fs) return;

      this.pgm = gl.createProgram();
      gl.attachShader(this.pgm, vs);
      gl.attachShader(this.pgm, fs);
      gl.linkProgram(this.pgm);
      if (!gl.getProgramParameter(this.pgm, gl.LINK_STATUS)) {
         console.error('Program link error:', gl.getProgramInfoLog(this.pgm));
         return;
      }
      gl.useProgram(this.pgm);

      // Get attribute locations from the compiled program
      const posLoc = gl.getAttribLocation(this.pgm, 'position');
      const uvLoc = gl.getAttribLocation(this.pgm, 'uv');

      // Position buffer: a single full-screen triangle in clip space.
      // (-1,-1) is bottom-left, (3,-1) extends far right, (-1,3) extends far up.
      // The GPU clips to the viewport, so the visible area is exactly [-1,1]².
      const posBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      // UV buffer: matching texture coordinates for the triangle.
      // (0,0) maps to the bottom-left corner; values > 1 are clipped away.
      const uvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 2, 0, 0, 2]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(uvLoc);
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);

      // Cache uniform locations so we can efficiently set them each frame
      this.u = {};
      ['iTime', 'iResolution', 'hue', 'hover', 'rot', 'hoverIntensity', 'backgroundColor'].forEach(name => {
         this.u[name] = gl.getUniformLocation(this.pgm, name);
      });

      // Enable standard alpha blending for transparent compositing
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0, 0, 0, 0);
   }

   /* =============================================================
      _resize()
      =============================================================
      Keeps the canvas resolution in sync with its on-screen size.
      CSS sizes the canvas element (100% × 100%), but the actual
      pixel buffer must be set explicitly via canvas.width/height.
      We multiply by devicePixelRatio so the orb looks sharp on
      HiDPI / Retina displays. The gl.viewport call tells WebGL
      to use the full buffer.
      ============================================================= */
   _resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      this.canvas.width = w * dpr;
      this.canvas.height = h * dpr;
      if (this.gl) this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
   }

   /* =============================================================
      _loop(ts)
      =============================================================
      The animation frame callback — called ~60 times per second by
      the browser via requestAnimationFrame.
      Each frame it:
      1. Schedules the next frame immediately (so animation never
         stops, even if this frame is slow).
      2. Converts the browser's millisecond timestamp to seconds and
         computes the delta-time (dt) since the last frame.
      3. Smoothly interpolates currentHover toward targetHover using
         an exponential ease (lerp with dt-scaled factor). This gives
         a nice fade-in / fade-out when setActive() is toggled.
      4. Accumulates rotation while active (currentHover > 0.5).
      5. Clears the canvas (transparent), uploads all uniform values
         for this frame, and issues a single draw call (3 vertices =
         one triangle that covers the screen).
      ============================================================= */
   _loop(ts) {
      this._raf = requestAnimationFrame(this._loop.bind(this));
      if (!this.pgm) return;
      const gl = this.gl;
      const t = ts * 0.001;                                        // ms → seconds
      const dt = this.lastTs ? t - this.lastTs : 0.016;           // delta time (fallback ~60fps)
      this.lastTs = t;

      // Continuous subtle rotation for organic life
      this.currentRot += dt * 0.08;

      // Smooth hover interpolation: exponential ease toward target
      this.currentHover += (this.targetHover - this.currentHover) * Math.min(dt * 4, 1);
      
      // Smoothly interpolate real-time audio amplitude
      this.audioAmp += (this.targetAudioAmp - this.audioAmp) * Math.min(dt * 15, 1);

      let speakBoost = 0;
      let hueShimmer = 0;

      if (this.speaking) {
         this.speakClock += dt;
         const w1 = Math.sin(this.speakClock * 7.3);
         const w2 = Math.sin(this.speakClock * 4.1 + 1.4);
         const w3 = Math.sin(this.speakClock * 11.7 + 0.6);
         const combined = (w1 * 0.5 + w2 * 0.35 + w3 * 0.25);
         // Blend algorithmic speak rhythm with real Web Audio amplitude
         speakBoost = Math.max(0, combined) * 0.25 + (this.audioAmp * 0.65);
         hueShimmer = (combined * 4) + (this.audioAmp * 15);   // Dynamic hue shimmer on audio peaks
         this.currentRot += dt * 0.3 * (1 + speakBoost * 1.5);

         // Scale container in real time with audio volume level!
         if (this.container) {
            const scale = 1 + (this.audioAmp * 0.08);
            this.container.style.transform = `translate(-50%, -50%) scale(${scale})`;
         }
      } else {
         this.speakClock = 0;
         this.audioAmp = 0;
         this.targetAudioAmp = 0;
         if (this.container) this.container.style.transform = 'translate(-50%, -50%)';
      }

      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.pgm);
      gl.uniform1f(this.u.iTime, t);                              // elapsed seconds
      gl.uniform3f(this.u.iResolution, this.canvas.width, this.canvas.height, this.canvas.width / this.canvas.height);
      gl.uniform1f(this.u.hue, this.hue + hueShimmer);            // palette rotation, shimmered while speaking
      gl.uniform1f(this.u.hover, Math.max(0.25, this.currentHover));            // baseline noise movement + active boost
      gl.uniform1f(this.u.rot, this.currentRot);                  // accumulated rotation
      gl.uniform1f(this.u.hoverIntensity, this.hoverIntensity + speakBoost);   // wave distortion strength, boosted by talk rhythm while speaking
      gl.uniform3f(this.u.backgroundColor, this.bgColor[0], this.bgColor[1], this.bgColor[2]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);                          // draw the single full-screen triangle
   }

   setAudioAmplitude(amp) {
      this.targetAudioAmp = Math.max(0, Math.min(1, amp));
   }

   triggerPulse() {
      const ctn = this.container;
      if (!ctn) return;
      ctn.classList.remove('orb-user-pulse');
      void ctn.offsetWidth;
      ctn.classList.add('orb-user-pulse');
      setTimeout(() => ctn.classList.remove('orb-user-pulse'), 600);
   }

   /* =============================================================
      setActive(active)
      =============================================================
      Toggles the orb between its idle and active (e.g. "speaking")
      states.
      - When active=true, targetHover is set to 1.0. Over the next
        few frames, _loop() will smoothly ramp currentHover up to 1,
        which makes the shader apply the wavy UV distortion and the
        rotation starts accumulating. The CSS class 'active' can be
        used to style the container (e.g. scale or glow via CSS).
      - When active=false, the reverse happens — the distortion and
        rotation smoothly fade out.
      ============================================================= */
   setActive(active) {
      this.targetHover = active ? 1.0 : 0.0;
      const ctn = this.container;
      if (active) ctn.classList.add('active');
      else ctn.classList.remove('active');
   }

   /* =============================================================
      setSpeaking(speaking)
      =============================================================
      Turns on/off the distinct "talking" animation rhythm (see the
      speakBoost/hueShimmer calculation in _loop()). This is separate
      from setActive() — a call can be active (processing) without
      speaking, or speaking without the general active distortion.
      In practice TYRON's script.js calls setActive(true) AND
      setSpeaking(true) together while audio is playing, then both
      false when the sentence finishes.
      ============================================================= */
   setSpeaking(speaking) {
      this.speaking = !!speaking;
      const ctn = this.container;
      if (speaking) ctn.classList.add('orb-speaking');
      else ctn.classList.remove('orb-speaking');
   }

   /* =============================================================
      destroy()
      =============================================================
      Cleans up all resources so the renderer can be safely removed:
      1. Cancels the pending animation frame.
      2. Removes the window resize listener.
      3. Detaches the <canvas> element from the DOM.
      4. Asks the browser to release the WebGL context and its GPU
         memory via the WEBGL_lose_context extension.
      Always call this when the orb is no longer needed (e.g. when
      navigating away from the page or unmounting a component).
      ============================================================= */
   destroy() {
      cancelAnimationFrame(this._raf);
      window.removeEventListener('resize', this._onResize);
      if (this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
      const ext = this.gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
   }
}