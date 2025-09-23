import cloudinary
import cloudinary.uploader
import os
import cloudinary.api

# Cấu hình Cloudinary (lấy từ Dashboard)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "darz5ai4r"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "712931137569911"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "P-j5dUO0anP4AxdndzbnqFyimgI")
)

def upload_image_to_cloudinary(image_url: str) -> str:
    """
    Upload ảnh lên Cloudinary từ URL hoặc file local.
    Trả về link HTTPS (secure_url).
    """
    try:
        # Upload (có thể truyền URL trực tiếp hoặc path tới file local)
        res = cloudinary.uploader.upload(image_url)

        # Lấy secure_url (link HTTPS public)
        return res.get("secure_url", "")
    except Exception as e:
        print("Upload error:", e)
        return ""


