import { Component } from '@angular/core';
import { Router } from '@angular/router';

import { ExerciseMenuItem } from '../../models/exercise.models';

@Component({
  selector: 'app-main-page',
  templateUrl: './main.page.html',
  styleUrl: './main.page.css',
})
export class MainPageComponent {
  protected readonly menuItems: ExerciseMenuItem[] = [
    {
      type: 'ejercicio1',
      label: 'Ejercicio 1',
      description: 'Clasificacion Iris con modelos supervisados.',
      enabled: true,
    },
    {
      type: 'ejercicio2',
      label: 'Ejercicio 2',
      description: 'Reservado para nuevo flujo IA.',
      enabled: true,
    },
    {
      type: 'ejercicio3',
      label: 'Ejercicio 3',
      description: 'K-Means + MLP con analisis comparativo.',
      enabled: true,
    },
    {
      type: 'ejercicio4',
      label: 'Ejercicio 4',
      description: 'Reservado para nuevo flujo IA.',
      enabled: true,
    },
    {
      type: 'ejercicio5',
      label: 'Ejercicio 5',
      description: 'Escenas CV con features RGB, HSV y textura.',
      enabled: true,
    },
    {
      type: 'ejercicio6',
      label: 'Ejercicio 6',
      description: 'Reservado para nuevo flujo IA.',
      enabled: true,
    },
  ];

  constructor(private readonly router: Router) {}

  protected openExercise(type: string, enabled: boolean): void {
    if (!enabled) {
      return;
    }

    void this.router.navigate(['/respuesta', type]);
  }
}
