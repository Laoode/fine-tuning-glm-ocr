# RECIPE-DB KIE Verification Report

## Summary

| Metric | Count |
|--------|-------|
| Total labels | 817 |
| Labels with schema issues | 0 |
| Missing OCR text | 0 |

## Schema Issues (fix priority)

✅ No schema issues found.

## Review Checklist

For each receipt in Label Studio, verify:
- [ ] `store_name` matches what's visually prominent at the top
- [ ] `store_location` is complete (not truncated by OCR)
- [ ] `store_contacts.type` is copied as-is (not classified)
- [ ] `items` have correct `item_name` (no SKU codes mixed in)
- [ ] `unit_price` does not include currency symbol
- [ ] `grand_total` matches the final TOTAL line
- [ ] `payment_method` is copied exactly (not translated)
- [ ] `currency` is empty if not printed on receipt

## Common Errors from Gemini

| Error pattern | How to fix |
|---------------|------------|
| Currency symbol in price field | Remove symbol, keep number only |
| store_contacts.type guessed (e.g., 'TEL' from unlabelled number) | Set type to '' |
| item_name includes barcode/SKU | Remove code, keep description only |
| total_discount calculated (summed) | Use only explicitly written value |
| payment_method translated (e.g., 'Cash' from 'Tunai') | Restore original |