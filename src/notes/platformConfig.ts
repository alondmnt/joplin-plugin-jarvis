import joplin from 'api';

export type RuntimePlatform = 'desktop' | 'mobile';

export interface RetrievalPlatformConfig {
  platform: RuntimePlatform;
  candidateLimit: number;
  pairwisePoolSize: number;
  pairwiseCap: number;
  topK: number;
  maxPassagesPerNote: number;
  windowTokens: number;
  maxAnswerPassages: number;
}

const DESKTOP_CONFIG: RetrievalPlatformConfig = {
  platform: 'desktop',
  candidateLimit: 40,
  pairwisePoolSize: 12,
  pairwiseCap: 20,
  topK: 8,
  maxPassagesPerNote: 2,
  windowTokens: 150,
  maxAnswerPassages: 6,
};

const MOBILE_CONFIG: RetrievalPlatformConfig = {
  platform: 'mobile',
  candidateLimit: 25,
  pairwisePoolSize: 8,
  pairwiseCap: 12,
  topK: 5,
  maxPassagesPerNote: 3,
  windowTokens: 140,
  maxAnswerPassages: 4,
};

/**
 * Detect the runtime platform via Joplin versionInfo and expose tuned retrieval limits.
 * Keeps Step B–D defaults in sync across desktop/mobile without scattering constants.
 */
export async function getRetrievalPlatformConfig(): Promise<RetrievalPlatformConfig> {
  try {
    const info = await joplin.versionInfo() as { platform?: RuntimePlatform };
    const platform: RuntimePlatform = info?.platform === 'mobile' ? 'mobile' : 'desktop';
    return platform === 'mobile' ? MOBILE_CONFIG : DESKTOP_CONFIG;
  } catch (error) {
    console.warn('Failed to resolve platform; defaulting to desktop config', error);
    return DESKTOP_CONFIG;
  }
}
