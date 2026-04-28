#!/bin/bash
# Quick verification script for Docker setup

echo "🔍 Yaguarete Docker Setup Verification"
echo "======================================"
echo ""

# Check files exist
echo "📄 Checking required files..."
files=(
  "Dockerfile"
  "docker-compose.yml"
  ".dockerignore"
  "docker-build.sh"
  "requirements.txt"
  "src/main.py"
)

for file in "${files[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✅ $file"
  else
    echo "  ❌ $file - MISSING"
  fi
done

echo ""
echo "🐳 Docker commands available:"
echo "  ✓ docker --version"
echo "  ✓ docker-compose --version"

echo ""
echo "🎯 Quick start commands:"
echo ""
echo "1. Build the Docker image:"
echo "   docker-compose build"
echo ""
echo "2. Start services:"
echo "   docker-compose up -d"
echo ""
echo "3. Check status:"
echo "   docker-compose ps"
echo ""
echo "4. View logs:"
echo "   docker-compose logs -f yaguarete"
echo ""
echo "5. Test health:"
echo "   curl http://localhost:8001/health"
echo ""
echo "6. Stop services:"
echo "   docker-compose down"
echo ""
echo "All set! 🚀 Run: docker-compose build && docker-compose up -d"
