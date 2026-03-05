import { Routes } from '@angular/router';

import { ExerciseResponsePageComponent } from './pages/exercise-response/exercise-response.page';
import { MainPageComponent } from './pages/main/main.page';

export const routes: Routes = [
  {
    path: '',
    component: MainPageComponent,
    title: 'Laboratorio IA',
  },
  {
    path: 'respuesta/:type',
    component: ExerciseResponsePageComponent,
    title: 'Resultado de ejercicio',
  },
  {
    path: '**',
    redirectTo: '',
  },
];
