export type ExerciseType =
  | 'ejercicio1'
  | 'ejercicio2'
  | 'ejercicio3'
  | 'ejercicio4'
  | 'ejercicio5'
  | 'ejercicio6';

export interface ExerciseRequest {
  type: string;
  options?: Record<string, unknown>;
}

export interface ImageArtifact {
  name: string;
  path: string;
  url?: string;
  content_base64?: string;
}

export interface ExerciseResponse {
  type: string;
  summary: Record<string, unknown>;
  images: ImageArtifact[];
}

export interface ExerciseMenuItem {
  label: string;
  type: ExerciseType;
  description: string;
  enabled: boolean;
}
