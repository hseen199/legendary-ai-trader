# وثائق API الإشعارات (Notifications API)

## نظرة عامة

نظام الإشعارات مصمم لإعلام المستخدمين بالأحداث المهمة المتعلقة بحساباتهم، مثل الإيداعات، السحوبات، وغيرها. يتم إرسال الإشعارات إلى الواجهة الأمامية ويمكن عرضها في صفحة الإشعارات `/notifications`.

---

## Endpoints

| الطريقة | المسار | الوصف |
|---|---|---|
| `GET` | `/api/v1/notifications` | الحصول على قائمة بجميع إشعارات المستخدم الحالي. |
| `GET` | `/api/v1/notifications/unread-count` | الحصول على عدد الإشعارات غير المقروءة. |
| `POST` | `/api/v1/notifications/{id}/read` | تحديد إشعار معين كمقروء. |
| `POST` | `/api/v1/notifications/read-all` | تحديد جميع الإشعارات كمقروءة. |
| `DELETE` | `/api/v1/notifications/{id}` | حذف إشعار معين. |

---

## كائن الإشعار (Notification Object)

يمثل كل إشعار كائن JSON بالتنسيق التالي:

```json
{
  "id": 123,
  "type": "withdrawal",
  "title": "تمت الموافقة على السحب",
  "message": "تمت الموافقة على طلب سحبك بمبلغ 100.00$. يرجى تأكيد السحب من بريدك الإلكتروني.",
  "is_read": false,
  "created_at": "2026-01-18T18:00:00Z",
  "data": {
    "amount": 100.00,
    "status": "approved",
    "to_address": "0x123...abc"
  }
}
```

### حقول الكائن الرئيسي

| الحقل | النوع | الوصف |
|---|---|---|
| `id` | `integer` | المعرف الفريد للإشعار. |
| `type` | `string` | نوع الإشعار (e.g., `deposit`, `withdrawal`). |
| `title` | `string` | عنوان الإشعار. |
| `message` | `string` | نص رسالة الإشعار. |
| `is_read` | `boolean` | `true` إذا تمت قراءة الإشعار، وإلا `false`. |
| `created_at` | `string` | تاريخ ووقت إنشاء الإشعار (ISO 8601). |
| `data` | `object` | كائن يحتوي على بيانات إضافية خاصة بنوع الإشعار. |

---

## كائن البيانات (Data Object)

يحتوي كائن `data` على معلومات سياقية تختلف باختلاف نوع الإشعار.

### 1. إشعارات الإيداع (`type: "deposit"`)

| الحقل | النوع | الوصف |
|---|---|---|
| `amount` | `number` | مبلغ الإيداع. |
| `status` | `string` | حالة الإيداع (`pending`, `confirming`, `completed`, `failed`, `expired`). |

**مثال:**
```json
{
  "amount": 500.00,
  "status": "completed"
}
```

### 2. إشعارات السحب (`type: "withdrawal"`)

| الحقل | النوع | الوصف |
|---|---|---|
| `amount` | `number` | مبلغ السحب. |
| `status` | `string` | حالة السحب (`approved`, `rejected`, `completed`, `pending`). |
| `to_address` | `string` | عنوان محفظة المستلم (يظهر عند الموافقة). |
| `tx_hash` | `string` | رقم معاملة البلوكتشين (يظهر عند الإتمام). |
| `rejection_reason` | `string` | سبب رفض طلب السحب (يظهر عند الرفض). |

**مثال (موافقة):**
```json
{
  "amount": 100.00,
  "status": "approved",
  "to_address": "0xAbc...123"
}
```

**مثال (رفض):**
```json
{
  "amount": 200.00,
  "status": "rejected",
  "rejection_reason": "رصيد غير كافٍ"
}
```

**مثال (إتمام):**
```json
{
  "amount": 150.00,
  "status": "completed",
  "tx_hash": "0xDef...456"
}
```
