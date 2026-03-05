import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { ExerciseRequest, ExerciseResponse } from '../models/exercise.models';

@Injectable({
  providedIn: 'root',
})
export class ExerciseApiService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = 'http://localhost:8000';

  executeExercise(payload: ExerciseRequest): Observable<ExerciseResponse> {
    return this.http.post<ExerciseResponse>(
      `${this.apiBaseUrl}/api/v1/exercises/execute`,
      payload,
    );
  }

  toImageUrl(imageUrlOrPath: string | undefined): string {
    if (!imageUrlOrPath) {
      return '';
    }

    if (imageUrlOrPath.startsWith('http://') || imageUrlOrPath.startsWith('https://')) {
      return imageUrlOrPath;
    }

    if (imageUrlOrPath.startsWith('/')) {
      return `${this.apiBaseUrl}${imageUrlOrPath}`;
    }

    return `${this.apiBaseUrl}/${imageUrlOrPath}`;
  }
}
