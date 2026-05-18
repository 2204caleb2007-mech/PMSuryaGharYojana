// =========================================================
// Shared TypeScript interfaces for the PM Surya Ghar Yojana app
// =========================================================

/** Detection bounding box returned by Roboflow */
export interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

/** A single panel detection result */
export interface Detection {
    bbox: BoundingBox;
    confidence: number;
    class: string;
}

/** Full response from /api/analyze-image */
export interface AnalyzeImageResponse {
    sample_id: string;
    analysis_timestamp: string;
    has_solar: boolean;
    confidence: number;
    panel_count: number;
    detections: Detection[];
    message: string;
    model_used: string;
    model_id: string;
    annotated_image_url?: string;
}

/** Full response from /api/analyze-location */
export interface AnalyzeLocationResponse {
    sample_id: string;
    latitude: number;
    longitude: number;
    address?: string;
    has_solar: boolean;
    confidence: number;
    panel_count: number;
    pv_area_sqm?: number;
    capacity_kw?: number;
    qc_status: string;
    model_used: string;
    satellite_image_source?: string;
    predictions: Detection[];
    saved_to_db: boolean;
    analysis_id?: number;
    // Fallback optional fields
    lat?: number;
    lon?: number;
    detection_count?: number;
    analysis_timestamp?: string;
    zoom_level?: number;
}

/** Response from /api/subsidy */
export interface SubsidyResponse {
    capacity: number;
    state: string;
    subsidy_range: string;
    min_subsidy: number;
    max_subsidy: number;
    avg_subsidy: number;
    currency: string;
    based_on_sample: string;
    panel_count: number;
}

/** Response from /api/health */
export interface HealthResponse {
    status: string;
    timestamp: string;
    models: {
        roboflow_inference: string;
        local_inference_model: string;
        retinanet_detector: string;
    };
    system_stats: {
        total_analyses: number;
        data_directory: string[];
        roboflow_model: string;
    };
    note: string;
}

/** Saved analysis record */
export interface SampleRecord {
    sample_id: string;
    has_solar: boolean;
    panel_count: number;
    confidence: number;
    timestamp: string;
    model_used: string;
}

/** QC Status values */
export type QCStatus = 'VERIFIED' | 'PARTIAL' | 'PENDING' | 'NOT VERIFIED';

/** Analysis UI state */
export interface AnalysisState {
    sampleId: string;
    hasSolar: boolean;
    panelCount: number;
    confidence: number;
    pvArea: string;
    capacityKw: number;
    qcStatus: QCStatus;
    qcNotes: string[];
    rawJson: string;
    isLoading: boolean;
    error: string | null;
}

/** Chat message for live chat */
export interface ChatMessage {
    sender: string;
    message: string;
    time: string;
    type: 'sent' | 'received' | 'system';
}

/** Benefit carousel slide */
export interface BenefitSlide {
    icon: string;
    iconColor: string;
    title: string;
    description: string;
}

/** Process step */
export interface ProcessStep {
    number: string;
    icon: string;
    title: string;
    description: string;
}
