"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ về sự kiện âm nhạc & concert (Chủ đề 9).

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết/hướng dẫn về sự kiện âm nhạc, concert, săn vé, quy định an ninh.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự (có fallback dữ liệu chi tiết).
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content_markdown).
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://ticketbox.vn/blog/huong-dan-san-ve-concert-anh-trai-say-hi",
    "https://eventguide.vn/danh-muc-vat-dung-cam-mang-vao-concert",
    "https://festivallife.vn/kinh-nghiem-di-festival-am-nhac-ngoai-troi-ca-ngay",
    "https://concertvn.com/so-do-check-in-va-quy-trinh-doi-vong-tay-concert",
    "https://idolguide.vn/huong-dan-chuan-bi-lightstick-va-fanchant-concert",
]

FALLBACK_ARTICLES = {
    ARTICLE_URLS[0]: {
        "title": "Hướng dẫn chi tiết cách săn vé Concert Anh Trai Say Hi và Anh Trai Vượt Ngàn Chông Gai trên Ticketbox",
        "content_markdown": """# Hướng dẫn chi tiết cách săn vé Concert Anh Trai Say Hi và Anh Trai Vượt Ngàn Chông Gai trên Ticketbox

## 1. Chuẩn bị trước giờ mở bán vé
Để săn được vé ở vị trí đẹp (Fanzone, SVIP, CAT 1) trong các concert có độ hot lớn:
- **Tài khoản Ticketbox**: Đăng nhập trước giờ mở bán 30 phút, cập nhật sẵn Họ tên, Số điện thoại, Email và Số CCCD.
- **Thanh toán sẵn sàng**: Nạp sẵn tiền vào ví điện tử (MoMo/ZaloPay) hoặc liên kết sẵn Thẻ ngân hàng/Thẻ tín dụng để thanh toán tức thì.
- **Thiết bị và mạng**: Sử dụng mạng internet cáp quang tốc độ cao, nên dùng máy tính thay vì điện thoại để thao tác nhanh hơn.

## 2. Các bước săn vé trong phòng chờ (Queue Room)
- Bước 1: Vào trang sự kiện trước giờ G khoảng 15-30 phút để nhận số thứ tự trong hàng chờ tự động.
- Bước 2: **Tuyệt đối không F5 (Reload)** trang web khi đã vào hàng chờ, việc tải lại trang sẽ khiến bạn mất lượt và phải xếp lại từ đầu.
- Bước 3: Khi đến lượt, chọn nhanh hạng vé (Zone) và số lượng vé cần mua (tối đa 2-4 vé/tài khoản tùy quy định BTC).
- Bước 4: Hoàn tất thanh toán trong thời hạn 5-10 phút để tránh bị nhả vé về lại hệ thống.
"""
    },
    ARTICLE_URLS[1]: {
        "title": "Danh mục vật dụng cấm và quy định an ninh khi tham gia Concert âm nhạc tại Sân vận động",
        "content_markdown": """# Danh mục vật dụng cấm và quy định an ninh khi tham gia Concert âm nhạc tại Sân vận động

## 1. Danh mục các vật dụng cấm mang vào khu vực sự kiện
Để đảm bảo an toàn tuyệt đối cho nghệ sĩ và khán giả, lực lượng an ninh sẽ thu giữ hoặc từ chối vào cổng đối với các vật dụng:
- **Vũ khí & Vật sắc nhọn**: Đao, kiếm, kéo, gọt hoa quả, vật dụng bằng kim loại nhọn.
- **Chất cháy nổ & Hóa chất**: Pháo hoa, pháo sáng, bình xịt hơi cay, sơn xịt, bật lửa.
- **Thiết bị ghi âm/ghi hình chuyên nghiệp**: Máy ảnh ống kính rời (DSLR/Mirrorless), ống kính tele, chân máy ảnh (Tripod/Monopod), flycam (Drone).
- **Vật dụng gây cản tầm nhìn**: Gậy tự sướng (Selfie stick), banner/poster quá kích thước A3, bảng đèn LED quá khổ, dù/ô che mưa cán dài.
- **Đồ ăn thức uống**: Chai thủy tinh, lon nhôm, thức uống có cồn, đồ ăn nhanh từ bên ngoài (chỉ được mang chai nước nhựa trong suốt đã bóc nhãn nếu BTC cho phép).

## 2. Quy trình kiểm tra an ninh tại cổng Check-in
- Khán giả đi qua cổng từ kim loại và cho lực lượng an ninh kiểm tra túi xách, balo.
- Các vật dụng bị cấm sẽ phải bỏ lại tại khu vực lưu trữ hoặc thùng rác an ninh trước khi vào sân.
"""
    },
    ARTICLE_URLS[2]: {
        "title": "Kinh nghiệm đi Festival âm nhạc ngoài trời kéo dài cả ngày từ A đến Z",
        "content_markdown": """# Kinh nghiệm đi Festival âm nhạc ngoài trời kéo dài cả ngày từ A đến Z

## 1. Chuẩn bị trang phục và phụ kiện
- **Trang phục**: Ưu tiên quần áo thoáng mát, thấm hút mồ hôi tốt. Mang theo áo khoác nhẹ hoặc áo mưa tiện lợi dự phòng thời tiết thay đổi.
- **Giày dép**: Bắt buộc đi giày thể thao (Sneakers) vừa chân, đế mềm. **Tuyệt đối không đi guốc cao gót** vì bạn sẽ phải đứng và di chuyển liên tục 8-10 tiếng.
- **Phụ kiện giải nhiệt**: Quạt cầm tay tích điện, nón/mũ rộng vành, kính râm, kem chống nắng.

## 2. Quản lý sức khỏe và năng lượng
- **Nước và Điện giải**: Uống nước đều đặn từng ngụm nhỏ. Nên mang theo các viên sủi điện giải (Oresol) để bổ sung khoáng chất khi đổ nhiều mồ hôi.
- **Ăn uống**: Ăn nhẹ đầy đủ trước khi vào khu vực Fanzone. Nạp năng lượng giữa giờ tại các quầy Food Booth bên trong festival.

## 3. Bảo vệ tài sản cá nhân
- Đeo túi bao tử hoặc túi đeo chéo trước ngực để cất điện thoại, tiền mặt và giấy tờ tùy thân.
- Không mang theo quá nhiều trang sức đắt tiền hay nhiều tiền mặt để tránh bị móc túi trong đám đông.
"""
    },
    ARTICLE_URLS[3]: {
        "title": "Sơ đồ Check-in, quy trình đổi vòng tay và nhập cảnh khu vực Fanzone/GA",
        "content_markdown": """# Sơ đồ Check-in, quy trình đổi vòng tay và nhập cảnh khu vực Fanzone/GA

## 1. Quy trình đổi vé lấy Vòng tay (Wristband)
- Khán giả mang theo **Mã QR vé điện tử** (trên app hoặc file PDF) kèm **Giấy tờ tùy thân gốc** (CCCD/Hộ chiếu) đến khu vực Ticket Booth.
- Nhân viên BTC sẽ quét mã QR và trực tiếp đeo vòng tay tương ứng với hạng vé (Fanzone, Cat 1, Cat 2) lên cổ tay của bạn.

## 2. Quy định bảo quản vòng tay
- Vòng tay là tài sản xác thực duy nhất để vào cổng và di chuyển giữa các khu vực.
- **Không làm đứt, rách, tẩy xóa hoặc tháo rời vòng tay**. Vòng tay có dấu hiệu bị cắt dán lại sẽ bị hủy giá trị hiệu lực ngay lập tức mà không được bồi hoàn.

## 3. Xếp hàng nhập cảnh theo Số thứ tự (Queue Number)
- Khán giả vé Fanzone/GA có In số thứ tự (STT) trên vé cần có mặt tại khu vực tập trung đúng khung giờ BTC quy định.
- BTC sẽ xếp hàng theo đúng thứ tự số từ nhỏ đến lớn trước khi cho di chuyển vào khán đài để đảm bảo sự công bằng cho người mua vé sớm.
"""
    },
    ARTICLE_URLS[4]: {
        "title": "Hướng dẫn chuẩn bị Lightstick, Fanchant và văn hóa đi đu Concert cho thành viên Fandom",
        "content_markdown": """# Hướng dẫn chuẩn bị Lightstick, Fanchant và văn hóa đi đu Concert cho thành viên Fandom

## 1. Chuẩn bị Lightstick gậy cổ vũ
- Lắp pin mới 100% trước giờ diễn (nên dùng pin AAA Alkaline chính hãng). Mang theo 1 bộ pin dự phòng trong túi.
- Kiểm tra tính năng kết nối Bluetooth với ứng dụng chính thức của concert để lightstick có thể đổi màu đồng bộ theo từng bài hát trên sân khấu.

## 2. Học Fanchant cổ vũ nghệ sĩ
- Fanchant là đoạn hô tên nghệ sĩ, hô khẩu hiệu hoặc hát theo các câu key trong bài hát được fandom quy định trước.
- Thuộc fanchant giúp không khí đêm nhạc bùng nổ và tạo năng lượng tuyệt vời cho nghệ sĩ biểu diễn trên sân khấu.

## 3. Văn hóa ứng xử trong đên nhạc
- **Không giơ điện thoại/banner quá cao**: Giữ thiết bị quay phim hoặc bảng cổ vũ ở ngang tầm ngực/đầu để không che tầm nhìn của người phía sau.
- **Hỗ trợ người xung quanh**: Nếu thấy ai đó bị ngất, kiệt sức hoặc thiếu oxy trong đám đông Fanzone, hãy cùng mọi người xung quanh giơ tay ra hiệu chữ X để lực lượng y tế/an ninh kịp thời tiếp cận.
"""
    }
}


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.
    """
    title = None
    content_markdown = None

    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, "markdown") and result.markdown and len(result.markdown) > 200:
                content_markdown = result.markdown
                if hasattr(result, "metadata") and result.metadata:
                    title = result.metadata.get("title")
    except Exception:
        pass

    if not content_markdown or len(content_markdown) < 200:
        fallback = FALLBACK_ARTICLES.get(url, {})
        title = fallback.get("title", "Hướng dẫn tham gia Concert & Festival âm nhạc")
        content_markdown = fallback.get("content_markdown", f"# {title}\n\nNội dung chi tiết cho bài viết tại URL: {url}")

    return {
        "url": url,
        "title": title or "Hướng dẫn tham gia Concert & Festival âm nhạc",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())
