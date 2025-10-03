import { useParams, useNavigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Product } from '@/types/product';
import { Button } from '@/components/ui/button';
import { ShoppingCart, ArrowLeft, Clock, Package } from 'lucide-react';
import { useCart } from '@/contexts/CartContext';
import { toast } from 'sonner';
import { getAllProducts } from '@/lib/api';

const ProductDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { addToCart } = useCart();
  const [product, setProduct] = useState<Product | null>(null);
  const [loading, setLoading] = useState(true);

  // Load real product data from API
  useEffect(() => {
    const loadProduct = async () => {
      setLoading(true);
      try {
        const products = await getAllProducts();
        const foundProduct = products.find(p => p.id.toString() === id);
        if (foundProduct) {
          setProduct(foundProduct);
        } else {
          toast.error('Product not found');
          navigate('/');
        }
      } catch (error) {
        console.error('Failed to load product:', error);
        toast.error('Failed to load product');
        navigate('/');
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      loadProduct();
    }
  }, [id, navigate]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Loading product...</p>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>Product not found</p>
      </div>
    );
  }

  const handleAddToCart = () => {
    addToCart(product);
    toast.success('Added to cart!');
  };

  const handleBuyNow = () => {
    addToCart(product);
    navigate('/checkout');
  };

  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-8">
        <Button
          variant="ghost"
          onClick={() => navigate(-1)}
          className="mb-6"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          {/* Image */}
          <div className="animate-fade-in">
            <div className="aspect-square rounded-lg overflow-hidden bg-muted">
              <img
                src={product.image_path}
                alt={product.name}
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Details */}
          <div className="flex flex-col animate-fade-in">
            <div className="mb-2">
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                {product.category}
              </span>
            </div>

            <h1 className="text-4xl font-bold mb-4">{product.name}</h1>

            <div className="text-3xl font-bold text-accent mb-6">
              ${product.price.toFixed(2)}
            </div>

            {/* Delivery Info */}
            <div className="space-y-3 mb-8 p-4 rounded-lg bg-muted/50">
              <div className="flex items-center space-x-3">
                <Clock className="h-5 w-5 text-accent" />
                <div>
                  <p className="font-medium">Delivery Time</p>
                  <p className="text-sm text-muted-foreground">{product.delivery}</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <Package className="h-5 w-5 text-accent" />
                <div>
                  <p className="font-medium">
                    {product.delivery === '30min' ? 'In Stock' : 'Custom Order'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {product.delivery === '30min'
                      ? 'Express delivery available'
                      : 'Made to order'}
                  </p>
                </div>
              </div>
            </div>

            {/* Description */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-3">Product Details</h2>
              <p className="text-muted-foreground leading-relaxed">
                Premium quality clothing designed for style and comfort. Made with
                high-quality materials and attention to detail. Perfect for everyday
                wear or special occasions.
              </p>
            </div>

            {/* Actions */}
            <div className="mt-auto space-y-3">
              <Button
                onClick={handleAddToCart}
                size="lg"
                className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
              >
                <ShoppingCart className="h-5 w-5 mr-2" />
                Add to Cart
              </Button>
              <Button
                onClick={handleBuyNow}
                size="lg"
                variant="outline"
                className="w-full"
              >
                Buy Now
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductDetail;
