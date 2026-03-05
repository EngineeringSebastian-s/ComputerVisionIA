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
}
