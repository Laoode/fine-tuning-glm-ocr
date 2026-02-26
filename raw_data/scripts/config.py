import json

EXTRACTION_SCHEMA_TEMPLATE = {
    "info": {
        "receipt_id": "",
        "store_name": "",
        "store_location": "",
        "payment_date": "",
        "payment_time": "",
    },
    "items": [
        {
            "item_name": "",
            "quantity": "",
            "unit_price": "",
            "total_price": "",
        }
    ],
    "payment": {
        "subtotal": "",
        "tax": "",
        "rounding": "",
        "discount": "",
        "voucher": "",
        "grand_total": "",
        "payment_method": "",
        "change": "",
    }
}