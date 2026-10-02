/**
 * The page background: near-black with a single, very deep-red bloom behind the
 * workspace.
 *
 * Static by design. Nothing here moves, loops or reacts to the pointer - a
 * screening tool should hold still while a clinician reads it. The glow is
 * painted as a radial gradient rather than a blurred box, so it costs nothing
 * to composite and has no edge for the eye to catch.
 *
 * The whole tree is `aria-hidden`, sits behind the app at `-z-10` and ignores
 * pointer events, so none of it is reachable by keyboard, announced by a screen
 * reader, or hit-tested by a click.
 */
export function AppBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-canvas"
    >
      {/* Main bloom, centred above the fold and kept low-contrast so it reads
          as depth rather than as a coloured panel. */}
      <div className="absolute -top-[42vh] left-1/2 h-[95vh] w-[120vw] -translate-x-1/2 bg-[radial-gradient(closest-side,rgb(127_29_29/0.28),rgb(69_10_18/0.10)_46%,transparent_74%)]" />

      {/* A single dim counter-light in the opposite corner, so the page is not
          uniformly warm. */}
      <div className="absolute -bottom-[28vh] -left-[12%] h-[70vh] w-[80vw] bg-[radial-gradient(closest-side,rgb(153_27_27/0.13),transparent_72%)]" />

      {/* Vignette: pulls the edges back to near-black so the cards sit on the
          darkest part of the canvas. */}
      <div className="absolute inset-0 bg-[radial-gradient(115%_90%_at_50%_0%,transparent_30%,rgb(5_5_7/0.5)_74%,rgb(5_5_7/0.82))]" />
    </div>
  )
}