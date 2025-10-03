import { useState, useEffect } from 'react';
import { Search } from 'lucide-react';
import ProductCard from '@/components/ProductCard';
import CategoryFilter from '@/components/CategoryFilter';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Product } from '@/types/product';
import { searchProducts, getAllProducts } from '@/lib/api';
import { toast } from 'sonner';
import heroImage from '@/assets/hero-fashion.jpg';

const Home = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Load real inventory from backend
    const loadInventory = async () => {
      setLoading(true);
      try {
        const inventoryProducts = await getAllProducts();
        
        if (inventoryProducts.length > 0) {
          setProducts(inventoryProducts);
          setFilteredProducts(inventoryProducts);
          toast.success(`Loaded ${inventoryProducts.length} items from inventory`);
        } else {
          toast.info('No inventory items found');
        }
      } catch (error) {
        console.error('Failed to load inventory:', error);
        toast.error('Failed to load inventory');
      } finally {
        setLoading(false);
      }
    };

    loadInventory();
  }, []);

  useEffect(() => {
    let filtered = products;

    if (selectedCategory) {
      filtered = filtered.filter((p) => p.category === selectedCategory);
    }

    if (searchQuery) {
      filtered = filtered.filter((p) =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredProducts(filtered);
  }, [selectedCategory, searchQuery, products]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const results = await searchProducts(searchQuery);
      if (results.length > 0) {
        setProducts(results);
        toast.success(`Found ${results.length} items`);
      } else {
        toast.info('No exact matches found. Showing catalog.');
      }
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search unavailable. Showing catalog.');
    } finally {
      setLoading(false);
    }
  };

  const categories = Array.from(new Set(products.map((p) => p.category)));

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-[60vh] md:h-[70vh] overflow-hidden">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/50 via-black/30 to-background" />
        </div>
        <div className="relative container mx-auto px-4 h-full flex items-center">
          <div className="max-w-2xl animate-fade-in">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
              Style Delivered,<br />Lightning Fast
            </h1>
            <p className="text-lg md:text-xl text-white/90 mb-8">
              Shop the latest fashion with 30-minute delivery or custom orders in 2-3 days
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="flex-1 flex gap-2">
                <Input
                  placeholder="Search for clothing..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  className="bg-white/95 backdrop-blur"
                />
                <Button
                  onClick={handleSearch}
                  disabled={loading}
                  className="bg-accent hover:bg-accent/90"
                >
                  <Search className="h-5 w-5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Products Section */}
      <section className="container mx-auto px-4 py-12">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Shop Our Collection</h2>
          <p className="text-muted-foreground">
            {filteredProducts.length} items available
          </p>
        </div>

        <CategoryFilter
          categories={categories}
          selectedCategory={selectedCategory}
          onSelectCategory={setSelectedCategory}
        />

        {loading ? (
          <div className="text-center py-16">
            <p className="text-muted-foreground">Loading inventory...</p>
          </div>
        ) : filteredProducts.length === 0 ? (
          <div className="text-center py-16">
            <p className="text-muted-foreground">No products found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredProducts.map((product) => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default Home;
