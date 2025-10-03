# Instagram Clothing Bot 🤖👕

An AI-powered Instagram bot that helps users find clothing items from photos and reels, with instant inventory matching and ordering capabilities.

## 🚀 Features

- **Instagram DM Integration**: Users can send photos/reels to your bot
- **AI Clothing Recognition**: Uses CLIP model for visual similarity search
- **Smart Inventory Matching**: Finds exact matches or similar items
- **Instant Ordering**: Direct users to checkout website
- **Delivery Options**: 30-minute delivery or 2-3 day custom orders

## 🏗️ Architecture

```
Instagram DM → Node.js Bot → Python AI Service → Checkout Website
```

- **Node.js Bot**: Handles Instagram DMs using instagram-private-api
- **Python AI Service**: CLIP-based image recognition and similarity search
- **Checkout Website**: Simple HTML/JS ordering interface

## 📋 Prerequisites

- Node.js (v14+)
- Python (v3.8+)
- Instagram test account (⚠️ Use test account, not main account!)

## 🛠️ Quick Setup

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd instagram-clothing-bot
   ./setup.sh
   ```

2. **Configure credentials**:
   ```bash
   # Edit .env file
   INSTAGRAM_USERNAME=your_test_account
   INSTAGRAM_PASSWORD=your_password
   ```

3. **Add product images**:
   - Add your product images to `inventory_images/` directory
   - Update `inventory.json` with your products

4. **Start services**:
   ```bash
   # Terminal 1: Start AI service
   python ai_service.py
   
   # Terminal 2: Start Instagram bot
   npm start
   ```

## 📱 How to Use

1. **User sends DM** to your bot with clothing photo
2. **Bot processes** image using AI
3. **Bot responds** with:
   - ✅ Exact match → Order link
   - 🔄 Similar items → Alternative options
   - ⏳ No match → Custom order option

## 🎯 Demo Flow

```
User: [Sends photo of blue jacket]
Bot: 🔍 Processing your image... Please wait!
Bot: ✅ Found exact match!
     Blue Denim Jacket - $89.99
     Order here: https://yourstore.com/product/1
```

## 📁 Project Structure

```
instagram-clothing-bot/
├── bot.js                 # Instagram bot (Node.js)
├── ai_service.py          # AI service (Python)
├── inventory.json         # Product database
├── checkout.html          # Checkout website
├── inventory_images/       # Product images
├── downloads/            # Temporary image storage
└── package.json          # Node.js dependencies
```

## 🔧 Configuration

### Inventory Setup
Edit `inventory.json` to add your products:
```json
{
  "id": 1,
  "name": "Blue Denim Jacket",
  "price": 89.99,
  "image_path": "inventory_images/denim_jacket.jpg",
  "url": "https://yourstore.com/product/1",
  "delivery": "30min"
}
```

### AI Model Settings
The bot uses OpenAI's CLIP model for image recognition. You can adjust similarity thresholds in `ai_service.py`:

```python
# Exact match threshold (0.8 = 80% similarity)
if similarities[exact_match_idx] > 0.8:

# Similar items threshold (0.3 = 30% similarity)
similar_indices = np.where(similarities > threshold)[0]
```

## 🚨 Important Notes

- **Use Test Account**: Instagram may flag/ban accounts using unofficial APIs
- **Rate Limiting**: Don't send too many messages too quickly
- **Image Quality**: Clear, well-lit photos work best
- **Product Images**: Use high-quality product photos for better matching

## 🛡️ Security

- Store credentials in `.env` file (never commit to git)
- Use HTTPS for checkout website
- Validate all user inputs
- Implement proper error handling

## 🚀 Deployment

For production deployment:

1. **Use official Instagram API** (requires approval)
2. **Deploy to cloud** (AWS, GCP, Heroku)
3. **Add database** (PostgreSQL, MongoDB)
4. **Implement payment** (Stripe, PayPal)
5. **Add monitoring** (logs, analytics)

## 🐛 Troubleshooting

### Bot not responding?
- Check Instagram credentials in `.env`
- Ensure AI service is running on port 5000
- Check console for error messages

### Poor image recognition?
- Use higher quality product images
- Adjust similarity thresholds
- Add more diverse inventory items

### Instagram login issues?
- Use 2FA disabled test account
- Check for Instagram security prompts
- Try logging in manually first

## 📈 Future Enhancements

- [ ] Real-time message webhooks
- [ ] Advanced clothing detection (YOLO, Detectron2)
- [ ] Size recommendation system
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] Mobile app integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - feel free to use for personal and commercial projects.

## ⚠️ Disclaimer

This is a prototype for demonstration purposes. Using unofficial Instagram APIs may violate Instagram's Terms of Service. For production use, implement the official Instagram Graph API.

---

**Happy coding! 🎉**

Built with ❤️ for the future of AI-powered shopping.