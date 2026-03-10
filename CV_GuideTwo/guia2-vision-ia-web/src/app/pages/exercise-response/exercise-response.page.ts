import { DatePipe, JsonPipe, TitleCasePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { catchError, map, of, switchMap } from 'rxjs';

import { ExerciseResponse, ImageArtifact } from '../../models/exercise.models';
import { ExerciseApiService } from '../../services/exercise-api.service';

interface SummaryRow {
  key: string;
  value: string;
}

interface Exercise1ScoreRow {
  model: string;
  accuracy: number;
}

interface ReportMetricRow {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

interface Exercise1ReportBlock {
  model: string;
  accuracy: number;
  rows: ReportMetricRow[];
}

interface KmeansSeriesRow {
  k: number;
  inertia: number;
  silhouette: number | null;
}

interface Exercise5ModelRow {
  model: string;
  cvF1Mean: number;
  cvF1Std: number;
  testAccuracy: number;
  testF1: number;
}

interface Exercise5ModeBlock {
  featureSet: string;
  bestModel: string;
  bestF1: number;
  table: Exercise5ModelRow[];
}

@Component({
  selector: 'app-exercise-response-page',
  imports: [RouterLink, JsonPipe, DatePipe, TitleCasePipe],
  templateUrl: './exercise-response.page.html',
  styleUrl: './exercise-response.page.css',
})
export class ExerciseResponsePageComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly exerciseApiService = inject(ExerciseApiService);

  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);
  protected readonly response = signal<ExerciseResponse | null>(null);
  protected readonly summaryRows = signal<SummaryRow[]>([]);
  protected readonly renderedAt = signal<Date | null>(null);

  ngOnInit(): void {
    this.route.paramMap
      .pipe(
        map((params) => params.get('type') ?? 'ejercicio1'),
        switchMap((type) => {
          this.loading.set(true);
          this.error.set(null);
          this.response.set(null);
          this.summaryRows.set([]);

          const payload =
            type === 'ejercicio5'
              ? {
                  type,
                  options: {
                    feature_modes: ['rgb', 'rgb+hsv', 'texture', 'rgb+texture', 'rgb+hsv+texture'],
                    resize: [128, 128],
                  },
                }
              : { type };

          return this.exerciseApiService.executeExercise(payload).pipe(
            catchError((err: unknown) => {
              const message =
                err && typeof err === 'object' && 'status' in err
                  ? `Error al ejecutar ${type}. Verifica API/CORS/red.`
                  : `Error inesperado al ejecutar ${type}.`;
              this.error.set(message);
              this.loading.set(false);
              return of(null);
            }),
          );
        }),
      )
      .subscribe((data) => {
        if (!data) {
          return;
        }

        this.response.set(data);
        this.summaryRows.set(this.flattenSummary(data.summary));
        this.renderedAt.set(new Date());
        this.loading.set(false);
      });
  }

  protected toImageUrl(image: ImageArtifact): string {
    if (image.url) {
      return this.exerciseApiService.toImageUrl(image.url);
    }

    if (image.path) {
      const staticPath = image.path.startsWith('app/output/')
        ? image.path.replace('app/output/', '/static/')
        : image.path;
      return this.exerciseApiService.toImageUrl(staticPath);
    }

    return '';
  }

  protected isType(type: string): boolean {
    return this.response()?.type === type;
  }

  protected formatMetric(value: unknown, decimals = 4): string {
    const n = this.toNumber(value);
    return Number.isFinite(n) ? n.toFixed(decimals) : '-';
  }

  protected exercise1BestModel(): string {
    return String(this.response()?.summary?.['best_model'] ?? '-');
  }

  protected exercise1BestAccuracy(): number {
    return this.toNumber(this.response()?.summary?.['best_accuracy']);
  }

  protected exercise1Scores(): Exercise1ScoreRow[] {
    const summary = this.response()?.summary;
    if (!summary) {
      return [];
    }

    const scores = this.toRecord(summary['scores']);
    return Object.entries(scores).map(([model, value]) => ({
      model,
      accuracy: this.toNumber(value),
    }));
  }

  protected exercise1Reports(): Exercise1ReportBlock[] {
    const summary = this.response()?.summary;
    if (!summary) {
      return [];
    }

    const reports = this.toRecord(summary['reports']);
    return Object.entries(reports).map(([model, reportRaw]) => {
      const report = this.toRecord(reportRaw);
      const rows: ReportMetricRow[] = Object.entries(report)
        .filter(([, metrics]) => this.isRecord(metrics))
        .map(([label, metrics]) => {
          const m = this.toRecord(metrics);
          return {
            label,
            precision: this.toNumber(m['precision']),
            recall: this.toNumber(m['recall']),
            f1: this.toNumber(m['f1-score']),
            support: this.toNumber(m['support']),
          };
        });

      return {
        model,
        accuracy: this.toNumber(report['accuracy']),
        rows,
      };
    });
  }

  protected kmeansSeries(): KmeansSeriesRow[] {
    const kmeans = this.toRecord(this.response()?.summary?.['kmeans']);
    const kValues = Array.isArray(kmeans['k_values']) ? kmeans['k_values'] : [];
    const inertias = Array.isArray(kmeans['inertias']) ? kmeans['inertias'] : [];
    const silhouettes = Array.isArray(kmeans['silhouettes']) ? kmeans['silhouettes'] : [];

    return kValues.map((k, index) => ({
      k: this.toNumber(k),
      inertia: this.toNumber(inertias[index]),
      silhouette:
        silhouettes[index] === null || silhouettes[index] === undefined
          ? null
          : this.toNumber(silhouettes[index]),
    }));
  }

  protected kmeansBestK(): number {
    return this.toNumber(this.toRecord(this.response()?.summary?.['kmeans'])['k_opt']);
  }

  protected kmeansAccuracy(): number {
    return this.toNumber(this.toRecord(this.response()?.summary?.['kmeans'])['accuracy']);
  }

  protected mlpBestCvAccuracy(): number {
    return this.toNumber(this.toRecord(this.response()?.summary?.['mlp'])['best_cv_accuracy']);
  }

  protected mlpTestAccuracy(): number {
    return this.toNumber(this.toRecord(this.response()?.summary?.['mlp'])['test_accuracy']);
  }

  protected mlpBestParams(): SummaryRow[] {
    const params = this.toRecord(this.toRecord(this.response()?.summary?.['mlp'])['best_params']);
    return Object.entries(params).map(([key, value]) => ({
      key,
      value: this.stringifyValue(value),
    }));
  }

  protected exercise5Modes(): Exercise5ModeBlock[] {
    const modesRaw = this.response()?.summary?.['modes'];
    if (!Array.isArray(modesRaw)) {
      return [];
    }

    return modesRaw.map((modeRaw) => {
      const mode = this.toRecord(modeRaw);
      const tableRaw = Array.isArray(mode['table']) ? mode['table'] : [];
      const table: Exercise5ModelRow[] = tableRaw.map((rowRaw) => {
        const row = this.toRecord(rowRaw);
        return {
          model: String(row['model'] ?? '-'),
          cvF1Mean: this.toNumber(row['cv_f1_macro_mean']),
          cvF1Std: this.toNumber(row['cv_f1_macro_std']),
          testAccuracy: this.toNumber(row['test_accuracy']),
          testF1: this.toNumber(row['test_f1_macro']),
        };
      });

      return {
        featureSet: String(mode['feature_set'] ?? '-'),
        bestModel: String(mode['best_model'] ?? '-'),
        bestF1: this.toNumber(mode['best_test_f1_macro']),
        table,
      };
    });
  }

  private flattenSummary(summary: Record<string, unknown>): SummaryRow[] {
    const rows: SummaryRow[] = [];

    const visit = (node: unknown, path: string): void => {
      if (node === null || node === undefined) {
        rows.push({ key: path, value: String(node) });
        return;
      }

      if (typeof node === 'object' && !Array.isArray(node)) {
        const entries = Object.entries(node as Record<string, unknown>);
        if (entries.length === 0) {
          rows.push({ key: path, value: '{}' });
          return;
        }

        for (const [key, value] of entries) {
          const nextPath = path ? `${path}.${key}` : key;
          visit(value, nextPath);
        }
        return;
      }

      if (Array.isArray(node)) {
        rows.push({ key: path, value: JSON.stringify(node) });
        return;
      }

      rows.push({ key: path, value: String(node) });
    };

    visit(summary, 'summary');
    return rows;
  }

  private toRecord(value: unknown): Record<string, unknown> {
    return this.isRecord(value) ? value : {};
  }

  private isRecord(value: unknown): value is Record<string, unknown> {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  private toNumber(value: unknown): number {
    if (typeof value === 'number') {
      return value;
    }
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    }
    return 0;
  }

  private stringifyValue(value: unknown): string {
    if (value === null || value === undefined) {
      return String(value);
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  }
}
