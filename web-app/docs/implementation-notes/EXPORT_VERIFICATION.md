# Export and Sharing Verification Guide

## Quick Verification Steps

### Backend Verification

1. **Install Dependencies**
   ```bash
   cd web-app/backend
   pip install -r requirements.txt
   ```

2. **Run Export Tests**
   ```bash
   python -m pytest tests/test_export.py -v
   ```
   
   Expected output: All 5 tests should pass
   - ✅ test_export_as_json
   - ✅ test_export_as_csv
   - ✅ test_pdf_export_service
   - ✅ test_pdf_export_with_minimal_data
   - ✅ test_json_export_includes_all_fields

3. **Check Syntax**
   ```bash
   python -m py_compile app/services/pdf_export_service.py
   python -m py_compile app/routers/evaluations.py
   ```

### Frontend Verification

1. **Check TypeScript Compilation**
   ```bash
   cd web-app/frontend
   npx tsc --noEmit --skipLibCheck src/utils/exportUtils.ts
   npx tsc --noEmit --skipLibCheck src/components/export/ExportMenu.tsx
   npx tsc --noEmit --skipLibCheck src/components/export/ShareButton.tsx
   npx tsc --noEmit --skipLibCheck src/pages/SharedEvaluationView.tsx
   ```

2. **Verify Component Structure**
   ```bash
   ls -la src/components/export/
   # Should show: ExportMenu.tsx, ShareButton.tsx, index.ts
   
   ls -la src/pages/
   # Should show: SharedEvaluationView.tsx
   
   ls -la src/utils/
   # Should show: exportUtils.ts
   ```

### Manual Testing (with running application)

#### Test JSON Export
1. Complete an evaluation in the web app
2. Click the "Export" button in the evaluation result card
3. Select "Export as JSON"
4. Verify file downloads as `evaluation_{session_id}.json`
5. Open the file and verify it contains:
   - Session metadata
   - Judge results with reasoning
   - Verifier verdicts
   - Confidence metrics
   - Hallucination metrics

#### Test CSV Export
1. Click the "Export" button
2. Select "Export as CSV"
3. Verify file downloads as `evaluation_{session_id}.csv`
4. Open in spreadsheet application
5. Verify sections:
   - Session summary
   - Judge results
   - Flagged issues
   - Verifier verdicts

#### Test PDF Export
1. Click the "Export" button
2. Select "Export as PDF"
3. Verify file downloads as `evaluation_{session_id}.pdf`
4. Open PDF and verify:
   - Title page with consensus score
   - Executive summary table
   - Judge results with reasoning
   - Confidence metrics
   - Hallucination analysis
   - Statistical metrics
   - Source text and candidate output

#### Test Share Functionality
1. Click the "Share" button in the evaluation result card
2. Verify modal appears with shareable link
3. Click "Copy" button
4. Verify "Copied!" message appears
5. Open new browser window/incognito mode
6. Paste the link (format: `http://localhost:3000/shared/{session_id}`)
7. Verify:
   - Page loads without authentication
   - "Read-Only" badge is visible
   - Yellow warning banner explains read-only access
   - Full evaluation results are displayed
   - Source text and candidate output are shown

### API Endpoint Testing (with curl)

#### Test JSON Export
```bash
# Replace {session_id} and {token} with actual values
curl -H "Authorization: Bearer {token}" \
  "http://localhost:8000/api/v1/evaluations/{session_id}/export?format=json" \
  -o evaluation.json
```

#### Test CSV Export
```bash
curl -H "Authorization: Bearer {token}" \
  "http://localhost:8000/api/v1/evaluations/{session_id}/export?format=csv" \
  -o evaluation.csv
```

#### Test PDF Export
```bash
curl -H "Authorization: Bearer {token}" \
  "http://localhost:8000/api/v1/evaluations/{session_id}/export?format=pdf" \
  -o evaluation.pdf
```

#### Test Share Link Creation
```bash
curl -X POST \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  "http://localhost:8000/api/v1/evaluations/{session_id}/share"
```

Expected response:
```json
{
  "session_id": "...",
  "shareable_url": "/shared/{session_id}",
  "full_url": "/shared/{session_id}",
  "created_at": "2024-..."
}
```

#### Test Shared Evaluation Access (no auth required)
```bash
curl "http://localhost:8000/api/v1/evaluations/shared/{session_id}"
```

## Expected Results

### All Tests Pass ✅
- Backend: 5/5 tests passing
- Frontend: TypeScript compilation successful
- No syntax errors

### Export Functionality ✅
- JSON: Complete structured data with all fields
- CSV: Tabular format with organized sections
- PDF: Professional multi-page report

### Share Functionality ✅
- Link generation works
- Copy to clipboard works
- Shared view accessible without authentication
- Read-only indicators present
- Full evaluation data visible

## Troubleshooting

### PDF Export Fails
- Ensure reportlab is installed: `pip install reportlab==4.0.7`
- Check Python version: Requires Python 3.8+

### Frontend TypeScript Errors
- Run: `npm install` to ensure all dependencies are installed
- Ignore d3-dispatch type errors (known issue with library types)

### Share Link Not Working
- Verify route is added in `src/routes/index.tsx`
- Check that SharedEvaluationView component is imported
- Ensure backend endpoint is accessible

### Export Button Not Visible
- Verify ExportMenu is imported in EvaluationResultMessage
- Check that evaluation has completed status
- Verify session_id is available in results

## Success Criteria

✅ All backend tests pass  
✅ All frontend components compile without errors  
✅ JSON export downloads with complete data  
✅ CSV export downloads with organized sections  
✅ PDF export generates professional report  
✅ Share button creates unique link  
✅ Copy to clipboard works  
✅ Shared view accessible without authentication  
✅ Shared view displays full evaluation results  
✅ Read-only indicators present in shared view  

## Notes

- PDF generation uses ReportLab library for professional formatting
- Shareable links use session UUIDs (secure, hard to guess)
- No expiration on shared links (can be revoked by deleting evaluation)
- Export functionality requires completed evaluation status
- Share functionality only works for completed evaluations
