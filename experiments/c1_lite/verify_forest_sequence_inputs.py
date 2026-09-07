"""Read-only revalidation of the frozen per-event evidence for composition."""
import json
from forest_sequence_inputs import load_certified_inputs

if __name__ == '__main__':
    data = load_certified_inputs()
    print(json.dumps({'status': 'PASS_FROZEN_CERTIFICATE_INPUT_BINDINGS',
        'events': len(data['events']), 'queries': len(data['queries']),
        'files': len(data['bindings'])}), flush=True)
