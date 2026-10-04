'use client'

import { commands } from './protocol'

import type { CanvasPagePreparation, Point, TransformFrame } from './protocol'

export type CanvasColor = [number, number, number, number]
export type CanvasStrokeKind = 'paint' | 'erase' | 'inpaint'
export type WorkspaceColor = [number, number, number]

export interface CanvasStroke {
  kind: CanvasStrokeKind
  layer: string | null
  point: Point
  diameter: number
  color?: CanvasColor
}

export interface Canvas {
  resize(width: number, height: number, dpr: number, background: WorkspaceColor): void
  setView(zoom: number, translation: [number, number]): void
  stageManifest(manifest: Uint8Array): { token: number; missing: string[] }
  installResource(resource: string, packet: Uint8Array): Promise<void>
  hasActiveManifest(manifest: Uint8Array): boolean
  activateFrame(token: number): boolean
  activatePage(page: string, expectedRevision: number): boolean
  cacheFrame(token: number, page: string): boolean
  clear(): void
  previewOpacity(element: string, opacity: number | null): void
  beginTransform(elements: TransformFrame[]): void
  updateTransform(elements: TransformFrame[]): void
  finishTransform(): void
  cancelTransform(): void
  beginStroke(stroke: CanvasStroke): void
  extendStroke(points: Point[]): void
  finishStroke(): void
  cancelStroke(): void
  sampleColor(point: Point): Promise<CanvasColor>
  dispose(): void
}

interface CanvasTransformFrame {
  element: string
  frame: {
    x: number
    y: number
    width: number
    height: number
    angleDegrees: number
  }
}

interface CanvasHandle {
  resize(width: number, height: number, dpr: number, background: Uint8Array): void
  setView(zoom: number, translationX: number, translationY: number): void
  stageManifest(manifest: Uint8Array): { token: number; missing: string[] }
  installResource(resource: string, packet: Uint8Array): Promise<void>
  activateFrame(token: number): boolean
  activatePage(page: string, expectedRevision: bigint): boolean
  cacheFrame(token: number): boolean
  clear(): void
  previewOpacity(element: string, opacity: number | null): void
  beginTransform(elements: CanvasTransformFrame[]): void
  updateTransform(sequence: number, elements: CanvasTransformFrame[]): void
  finishTransform(): unknown
  cancelTransform(): void
  beginStroke(
    kind: CanvasStrokeKind,
    layer: string | null,
    point: Point,
    diameter: number,
    color: Uint8Array,
  ): void
  extendStroke(points: Point[]): void
  finishStroke(): unknown
  cancelStroke(): void
  sampleColor(x: number, y: number): Promise<Uint8Array>
  setDeviceLostCallback(callback: ((reason: string) => void) | null): void
  dispose(): void
  free(): void
}

interface CanvasModule {
  default(): Promise<unknown>
  createCanvas(element: HTMLCanvasElement): Promise<CanvasHandle>
}

let modulePromise: Promise<CanvasModule> | null = null
let activeCanvas: Canvas | null = null
let prefetchGeneration = 0

export async function createCanvas(
  element: HTMLCanvasElement,
  onDeviceLost: (reason: string) => void,
): Promise<Canvas> {
  if (!(await browserWebGpuAvailable())) {
    console.warn('WebGPU adapter unavailable; using server-rendered software canvas.')
    return createSoftwareCanvas(element)
  }

  try {
    const module = await loadCanvasModule()
    const canvas = await module.createCanvas(element)

    let transformSequence = 0
    let disposed = false
    let stagedManifest: { token: number; bytes: Uint8Array } | null = null
    let activeManifest: Uint8Array | null = null
    const cachedManifests = new Map<string, Uint8Array>()

    canvas.setDeviceLostCallback(onDeviceLost)

    return {
      resize: (width, height, dpr, background) =>
        canvas.resize(width, height, dpr, new Uint8Array([...background, 255])),

      setView: (zoom, translation) =>
        canvas.setView(zoom, translation[0], translation[1]),

      stageManifest: (manifest) => {
        const staged = canvas.stageManifest(manifest)
        stagedManifest = {
          token: staged.token,
          bytes: manifest.slice(),
        }
        return staged
      },

      installResource: (resource, packet) =>
        canvas.installResource(resource, packet),

      hasActiveManifest: (manifest) =>
        activeManifest !== null && equalBytes(activeManifest, manifest),

      activateFrame: (token) => {
        const activated = canvas.activateFrame(token)

        if (activated && stagedManifest?.token === token) {
          activeManifest = stagedManifest.bytes
          stagedManifest = null
        }

        return activated
      },

      activatePage: (page, expectedRevision) => {
        const activated = canvas.activatePage(page, BigInt(expectedRevision))

        if (activated) {
          activeManifest = cachedManifests.get(page) ?? null
        }

        return activated
      },

      cacheFrame: (token, page) => {
        const cached = canvas.cacheFrame(token)

        if (stagedManifest?.token === token) {
          if (cached) {
            cachedManifests.set(page, stagedManifest.bytes)
          }
          stagedManifest = null
        }

        return cached
      },

      clear: () => {
        canvas.clear()
        stagedManifest = null
        activeManifest = null
        cachedManifests.clear()
      },

      previewOpacity: (element, opacity) =>
        canvas.previewOpacity(element, opacity),

      beginTransform: (elements) => {
        transformSequence = 0
        canvas.beginTransform(canvasTransformFrames(elements))
      },

      updateTransform: (elements) =>
        canvas.updateTransform(
          ++transformSequence,
          canvasTransformFrames(elements),
        ),

      finishTransform: () => void canvas.finishTransform(),

      cancelTransform: () => canvas.cancelTransform(),

      beginStroke: (stroke) =>
        canvas.beginStroke(
          stroke.kind,
          stroke.layer,
          stroke.point,
          stroke.diameter,
          new Uint8Array(stroke.color ?? [0, 0, 0, 0]),
        ),

      extendStroke: (points) => canvas.extendStroke(points),

      finishStroke: () => void canvas.finishStroke(),

      cancelStroke: () => canvas.cancelStroke(),

      sampleColor: async (point) => {
        const color = await canvas.sampleColor(point.x, point.y)

        if (color.length !== 4) {
          throw new Error('The WebGPU canvas returned an invalid color.')
        }

        return [color[0], color[1], color[2], color[3]]
      },

      dispose: () => {
        if (disposed) return

        disposed = true

        try {
          canvas.setDeviceLostCallback(null)
        } finally {
          try {
            canvas.dispose()
          } finally {
            canvas.free()
          }
        }
      },
    }
  } catch (error) {
    console.warn(
      'WebGPU canvas initialization failed; using server-rendered software canvas.',
      error,
    )

    return createSoftwareCanvas(element)
  }
}

async function browserWebGpuAvailable(): Promise<boolean> {
  if (typeof navigator === 'undefined') {
    return false
  }

  const gpu = (
    navigator as Navigator & {
      gpu?: {
        requestAdapter?: (options?: {
          powerPreference?: 'low-power' | 'high-performance'
        }) => Promise<unknown>
      }
    }
  ).gpu

  if (!gpu?.requestAdapter) {
    return false
  }

  try {
    const adapter = await gpu.requestAdapter({
      powerPreference: 'low-power',
    })

    return adapter !== null && adapter !== undefined
  } catch {
    return false
  }
}

function createSoftwareCanvas(element: HTMLCanvasElement): Canvas {
  const context = element.getContext('2d')

  if (!context) {
    throw new Error('Canvas 2D context unavailable.')
  }

  let disposed = false
  let zoom = 1
  let translation: [number, number] = [0, 0]
  let dpr = 1
  let background: WorkspaceColor = [245, 245, 245]
  let image: ImageBitmap | null = null
  let requestId = 0
  let token = 0

  const redraw = () => {
    if (disposed) return

    const width = element.width / dpr
    const height = element.height / dpr

    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, width, height)

    context.fillStyle = `rgb(${background[0]}, ${background[1]}, ${background[2]})`
    context.fillRect(0, 0, width, height)

    if (!image) {
      return
    }

    context.save()
    context.translate(translation[0], translation[1])
    context.scale(zoom, zoom)
    context.drawImage(image, 0, 0)
    context.restore()
  }

  const loadPage = async (page: string, revision: number) => {
    const currentRequest = ++requestId

    try {
      const response = await fetch('/rpc/get_canvas_preview', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          page,
          revision,
        }),
      })

      if (!response.ok) {
        throw new Error(
          `Canvas preview request failed with HTTP ${response.status}`,
        )
      }

      const blob = await response.blob()
      const nextImage = await createImageBitmap(blob)

      if (disposed || currentRequest !== requestId) {
        nextImage.close()
        return
      }

      image?.close()
      image = nextImage
      redraw()
    } catch (error) {
      console.error('Failed to load server-rendered canvas preview.', error)
    }
  }

  const nextToken = () => {
    token = token >= 0x7fffffff ? 1 : token + 1
    return token
  }

  return {
    resize: (width, height, nextDpr, nextBackground) => {
      dpr = Math.max(1, nextDpr)
      background = [
        nextBackground[0] ?? 245,
        nextBackground[1] ?? 245,
        nextBackground[2] ?? 245,
      ]

      element.width = Math.max(1, Math.round(width * dpr))
      element.height = Math.max(1, Math.round(height * dpr))

      redraw()
    },

    setView: (nextZoom, translationTuple) => {
      zoom = Number.isFinite(nextZoom) && nextZoom > 0 ? nextZoom : 1
      translation = [translationTuple[0], translationTuple[1]]
      redraw()
    },

    stageManifest: (_manifest) => ({
      token: nextToken(),
      missing: [] as string[],
    }),

    installResource: async (_resource, _packet) => {
      // Resources are already composed server-side into the preview image.
    },

    hasActiveManifest: (_manifest) => false,

    activateFrame: (_token) => {
      redraw()
      return true
    },

    activatePage: (page, expectedRevision) => {
      const revision = Number(expectedRevision)

      if (Number.isFinite(revision)) {
        void loadPage(page, revision)
      }

      return true
    },

    cacheFrame: (_token, _page) => true,

    clear: () => {
      image?.close()
      image = null
      redraw()
    },

    previewOpacity: (_element, _opacity) => {
      // The complete frame is server-rendered in software mode.
    },

    beginTransform: (_elements) => {
      // Backend commit remains functional; live GPU preview is unavailable.
    },

    updateTransform: (_elements) => {
      // Backend commit remains functional; live GPU preview is unavailable.
    },

    finishTransform: () => undefined,

    cancelTransform: () => undefined,

    beginStroke: (_stroke) => {
      // Backend commit remains functional; live GPU preview is unavailable.
    },

    extendStroke: (_points) => undefined,

    finishStroke: () => undefined,

    cancelStroke: () => undefined,

    sampleColor: async (point) => {
      const x = Math.max(
        0,
        Math.min(
          element.width - 1,
          Math.floor(point.x * dpr),
        ),
      )

      const y = Math.max(
        0,
        Math.min(
          element.height - 1,
          Math.floor(point.y * dpr),
        ),
      )

      const data = context.getImageData(x, y, 1, 1).data
      return [data[0], data[1], data[2], data[3]]
    },

    dispose: () => {
      if (disposed) return

      disposed = true
      requestId++

      image?.close()
      image = null
    },
  }
}

function loadCanvasModule(): Promise<CanvasModule> {
  // @ts-ignore -- wasm-pack output is created by bridge build tasks and absent in clean checkouts.
  modulePromise ??= import('./wasm/koharu_canvas.js')
    .then(async (module) => {
      const canvas = module as CanvasModule
      await canvas.default()
      return canvas
    })
    .catch((error: unknown) => {
      modulePromise = null
      throw error
    })
  return modulePromise
}

export async function fetchCanvasManifest(generation: number): Promise<Uint8Array> {
  return canvasBytesView(await commands.getCanvasManifest(generation))
}

export async function fetchCanvasResource(
  generation: number,
  resource: string,
): Promise<Uint8Array> {
  return canvasBytesView(await commands.getCanvasResource(generation, resource))
}

// Specta exposes CanvasBytes as number[], while its IpcResponse sends an ArrayBuffer at runtime.
function canvasBytesView(bytes: number[] | ArrayBuffer): Uint8Array {
  return new Uint8Array(bytes)
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  return (
    left.byteLength === right.byteLength && left.every((value, index) => value === right[index])
  )
}

export function activateCanvas(canvas: Canvas): () => void {
  prefetchGeneration++
  activeCanvas = canvas
  return () => {
    if (activeCanvas === canvas) {
      prefetchGeneration++
      activeCanvas = null
    }
  }
}

export function previewCanvasOpacity(element: string, opacity: number | null): void {
  activeCanvas?.previewOpacity(element, opacity)
}

export function showCanvasPage(page: string, expectedRevision: number | null): boolean {
  prefetchGeneration++
  return expectedRevision === null
    ? false
    : (activeCanvas?.activatePage(page, expectedRevision) ?? false)
}

export function cancelCanvasPrefetch(): void {
  prefetchGeneration++
}

export async function prefetchCanvasPages(pages: string[]) {
  const canvas = activeCanvas
  if (!canvas || pages.length === 0) return []
  const preparedPages: CanvasPagePreparation[] = []
  const generation = ++prefetchGeneration
  const current = () => activeCanvas === canvas && prefetchGeneration === generation
  for (const page of pages) {
    const prepared = await commands.prepareCanvasPage(page)
    if (!current() || prepared === null) return preparedPages
    const revision = prepared.revision
    const manifest = canvasBytesView(await commands.getCanvasPageManifest(page, revision))
    if (!current()) return preparedPages
    const staged = canvas.stageManifest(manifest)
    for (let offset = 0; offset < staged.missing.length; offset += 4) {
      const resources = staged.missing.slice(offset, offset + 4)
      const packets = await Promise.all(
        resources.map(
          async (resource) =>
            [
              resource,
              canvasBytesView(await commands.getCanvasPageResource(page, revision, resource)),
            ] as const,
        ),
      )
      if (!current()) return preparedPages
      await Promise.all(
        packets.map(([resource, packet]) => canvas.installResource(resource, packet)),
      )
    }
    if (!current()) return preparedPages
    if (canvas.cacheFrame(staged.token, page)) preparedPages.push(prepared)
  }
  return preparedPages
}

export function workspaceColor(): WorkspaceColor {
  const raw = window
    .getComputedStyle(document.documentElement)
    .getPropertyValue('--workspace-background-rgb')
  const values = raw.trim().split(/\s+/).map(Number)
  if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
    return [183, 180, 174]
  }
  return values.map((value) => Math.min(255, Math.max(0, Math.round(value)))) as WorkspaceColor
}

function canvasTransformFrames(elements: TransformFrame[]): CanvasTransformFrame[] {
  return elements.map(({ element, frame }) => ({
    element,
    frame: {
      x: frame.x,
      y: frame.y,
      width: frame.width,
      height: frame.height,
      angleDegrees: frame.angle_degrees,
    },
  }))
}
