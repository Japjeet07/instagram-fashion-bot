import { Product } from '@/types/product';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ShoppingCart, Clock } from 'lucide-react';
import { useCart } from '@/contexts/CartContext';
import { Link } from 'react-router-dom';

interface ProductCardProps {
  product: Product;
}

const ProductCard = ({ product }: ProductCardProps) => {
  const { addToCart } = useCart();


  const handleAddToCart = (e: React.MouseEvent) => {
    e.preventDefault();
    addToCart(product);
  };

  return (
    <Link to={`/product/${product.id}`}>
      <Card className="group overflow-hidden border-border hover:shadow-lg transition-all duration-300 animate-fade-in">
        <div className="relative aspect-square overflow-hidden bg-muted">
          <img
            src={product.image_path}
            alt={product.name}
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
            loading="lazy"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          
          {/* Delivery badge */}
          <div className="absolute top-3 left-3">
            <div className="flex items-center space-x-1 rounded-full bg-background/90 backdrop-blur px-3 py-1.5 text-xs font-medium">
              <Clock className="h-3 w-3 text-accent" />
              <span>{product.delivery}</span>
            </div>
          </div>

          {/* Quick add button */}
          <Button
            onClick={handleAddToCart}
            size="sm"
            className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-accent hover:bg-accent/90"
          >
            <ShoppingCart className="h-4 w-4 mr-1" />
            Add
          </Button>
        </div>

        <div className="p-4">
          <div className="mb-1">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {product.category}
            </span>
          </div>
          <h3 className="font-semibold text-foreground mb-2 line-clamp-2 group-hover:text-accent transition-colors">
            {product.name}
          </h3>
          <div className="flex items-center justify-between">
            <span className="text-lg font-bold text-primary">
              ${product.price.toFixed(2)}
            </span>
            {product.similarity_score && (
              <span className="text-xs text-muted-foreground">
                {Math.round(product.similarity_score * 100)}% match
              </span>
            )}
          </div>
        </div>
      </Card>
    </Link>
  );
};

export default ProductCard;
