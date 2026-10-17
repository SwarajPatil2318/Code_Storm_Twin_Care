"""Final integration preservation; historical scientific/UI snapshots stay immutable."""
import json
from pathlib import Path
from src.metabolic_d2.protocol import digest,verify_frozen
AUTHORIZED_UI_EDITS={
 'frontend/src/Sources.tsx','frontend/src/CardiovascularApp.tsx','frontend/src/Heart.tsx',
 'frontend/src/metabolic/MetabolicApp.tsx','frontend/src/metabolic/Visuals.tsx',
 'frontend/src/metabolic/ProductEvidence.tsx','frontend/src/metabolic/types.ts','frontend/src/metabolic/provenance.ts','frontend/src/metabolic/useLiveTwin.ts','frontend/src/metabolic/LiveDetails.tsx'}
AUTHORIZED_UI_ADDITIONS={'frontend/src/platform/WearableMiniCard.tsx','frontend/src/platform/useReceivedSources.ts',
                         'frontend/src/platform/wearable.css','frontend/src/metabolic/RiskDetails.tsx'}
def preservation():
 before=json.loads(Path('data/metabolic_risk_v1/integration_preservation_before.json').read_text())
 bad=[p for p,h in before.items() if p not in AUTHORIZED_UI_EDITS and (not Path(p).exists() or digest(p)!=h)]
 for p in Path('frontend/src').rglob('*'):
  if p.is_file() and str(p) not in before and str(p) not in AUTHORIZED_UI_ADDITIONS:bad.append(str(p))
 if bad:raise RuntimeError('Unexpected changes outside authorized integration scope')
 verify_frozen()
 return {'production_model_hashes_unchanged':True,'UI_changes_within_authorized_scope':True,
         'private_artifacts_unchanged':True,'spent_holdout_rows_decoded':0,'protected_files':len(before),
         'authorized_ui_edits':sorted(AUTHORIZED_UI_EDITS),'authorized_ui_additions':sorted(AUTHORIZED_UI_ADDITIONS)}
