import { DR_CLASSES } from '../../data/clinical'
import { LayersIcon } from '../icons'
import { Card, CardBody, CardHeader } from '../ui/Card'
import { ProbabilityBars } from './ProbabilityBars'

export interface ProbabilityCardProps {
  /**
   * Per-class softmax probabilities. `null` until a model actually runs, in
   * which case every bar renders as an empty hatched track.
   */
  probabilities: Record<string, number> | null
  predictedIndex: number | null
  /** Softmax sum check, shown so drift in the head is visible later. */
  probabilitySum: number | null
}

/**
 * Class-probability panel: one bar per DR class in the model's logit order
 * (`DR_CLASSES` mirrors `backend/inference/predictor.py::DR_CLASSES`).
 */
export function ProbabilityCard({
  probabilities,
  predictedIndex,
  probabilitySum,
}: ProbabilityCardProps) {
  const topClass =
    predictedIndex === null
      ? null
      : (DR_CLASSES.find((c) => c.index === predictedIndex) ?? null)

  return (
    <Card>
      <CardHeader
        icon={<LayersIcon />}
        title="Class Probabilities"
        subtitle="Softmax output of the 5-way DR classification head"
      />

      <CardBody className="space-y-4">
        <ProbabilityBars
          probabilities={probabilities}
          predictedIndex={predictedIndex}
        />

        <div className="flex items-center justify-between gap-3 border-t border-line pt-3 text-[11px]">
          <span className="text-ink-500">
            Top class
            <span className="ml-1.5 font-mono text-ink-400">
              {topClass ? `#${topClass.index}` : '-'}
            </span>
          </span>
          <span className="text-ink-500">
            Sum
            <span className="ml-1.5 font-mono text-ink-400">
              {probabilitySum === null ? '-' : probabilitySum.toFixed(4)}
            </span>
          </span>
        </div>
      </CardBody>
    </Card>
  )
}
